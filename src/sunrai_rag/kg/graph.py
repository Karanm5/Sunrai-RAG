"""Knowledge graph construction and graph-enhanced retrieval.

NetworkX is used rather than a graph database. At this scale a server-backed
store adds operational burden without changing any result, and an in-process
graph serialises to a single JSON file -- which keeps the reproduction a
one-command affair. The `expand` interface is storage-agnostic, so moving to
Neo4j later is a backend swap, not a redesign.

What the graph buys the retriever: a query mentioning a *method* can reach
the region reporting its *result* even when the two share no vocabulary --
the failure mode that pure embedding similarity handles worst.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx

from .extract import Entity, Extraction, Relation


class KnowledgeGraph:
    """Entity graph with provenance on every node and edge."""

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.graph = graph if graph is not None else nx.MultiDiGraph()

    # -- construction ------------------------------------------------------

    @staticmethod
    def from_extraction(extraction: Extraction) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        for entity in extraction.entities:
            kg.add_entity(entity)
        for relation in extraction.relations:
            kg.add_relation(relation)
        return kg

    def add_entity(self, entity: Entity) -> None:
        node = entity.node_id
        if self.graph.has_node(node):
            regions = self.graph.nodes[node].setdefault("source_region_ids", [])
            if entity.source_region_id not in regions:
                regions.append(entity.source_region_id)
        else:
            self.graph.add_node(
                node,
                name=entity.name,
                entity_type=entity.entity_type,
                source_region_ids=[entity.source_region_id],
            )

    def add_relation(self, relation: Relation) -> None:
        # Edges to undeclared nodes would create phantom entities with no
        # provenance, so they are dropped rather than auto-created.
        if not self.graph.has_node(relation.head) or not self.graph.has_node(
            relation.tail
        ):
            return
        self.graph.add_edge(
            relation.head,
            relation.tail,
            key=relation.relation_type,
            relation_type=relation.relation_type,
            source_region_id=relation.source_region_id,
        )

    # -- inspection --------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def stats(self) -> dict:
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
        rel_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rtype = data.get("relation_type", "unknown")
            rel_counts[rtype] = rel_counts.get(rtype, 0) + 1
        return {
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "entity_types": dict(sorted(type_counts.items())),
            "relation_types": dict(sorted(rel_counts.items())),
        }

    def regions_for_node(self, node_id: str) -> list[str]:
        if not self.graph.has_node(node_id):
            return []
        return list(self.graph.nodes[node_id].get("source_region_ids", []))

    # -- entity linking ----------------------------------------------------

    def link_query_entities(self, query: str, max_entities: int = 5) -> list[str]:
        """Find graph nodes mentioned in a query string.

        Deliberately simple: normalised substring matching on entity names,
        longest-first so "random forest classifier" is preferred over
        "random forest". A learned entity linker would raise recall but adds
        a component that cannot be validated on this dataset, so the simpler
        method is used and its limits are stated.
        """
        normalised = re.sub(r"\s+", " ", query.lower())
        matches: list[tuple[int, str]] = []
        for node, data in self.graph.nodes(data=True):
            name = str(data.get("name", "")).strip().lower()
            if len(name) < 3:
                continue
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", normalised):
                matches.append((len(name), node))
        matches.sort(key=lambda pair: (-pair[0], pair[1]))
        return [node for _, node in matches[:max_entities]]

    # -- expansion ---------------------------------------------------------

    def expand(self, seed_nodes: Sequence[str], hops: int = 1) -> set[str]:
        """Nodes reachable within `hops` edges of any seed, treating edges as
        undirected (a result is as reachable from its method as the reverse).

        Unknown seeds are ignored rather than raising, so a query mentioning
        nothing in the graph degrades to plain retrieval.
        """
        if hops < 0:
            raise ValueError("hops must be non-negative")
        present = [n for n in seed_nodes if self.graph.has_node(n)]
        if not present:
            return set()

        undirected = self.graph.to_undirected(as_view=True)
        reached: set[str] = set(present)
        frontier: set[str] = set(present)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(undirected.neighbors(node))
            next_frontier -= reached
            if not next_frontier:
                break
            reached.update(next_frontier)
            frontier = next_frontier
        return reached

    def expansion_paths(
        self, seed_nodes: Sequence[str], hops: int = 1, max_paths: int = 5
    ) -> list[list[str]]:
        """Concrete seed -> neighbour paths, for the explainability payload.

        These are what the demo renders as "the graph connected X to Y",
        turning the KG contribution into something a human can audit.
        """
        undirected = self.graph.to_undirected(as_view=True)
        paths: list[list[str]] = []
        for seed in seed_nodes:
            if not self.graph.has_node(seed):
                continue
            lengths = nx.single_source_shortest_path(undirected, seed, cutoff=hops)
            for _target, path in sorted(lengths.items()):
                if len(path) > 1:
                    paths.append(list(path))
                if len(paths) >= max_paths:
                    return paths
        return paths

    def regions_from_expansion(
        self, seed_nodes: Sequence[str], hops: int = 1, max_regions: int = 5
    ) -> list[str]:
        """Region ids reachable through the graph from the query's entities.

        Ordering is deterministic: nodes sorted by id, regions in insertion
        order, so the same query always yields the same evidence set.
        """
        reached = self.expand(seed_nodes, hops=hops)
        seeds = set(seed_nodes)
        # Prefer regions from expanded (non-seed) nodes -- those are the ones
        # plain retrieval would have missed, which is the KG's actual value.
        ordered_nodes = sorted(reached - seeds) + sorted(reached & seeds)
        regions: list[str] = []
        for node in ordered_nodes:
            for region_id in self.regions_for_node(node):
                if region_id not in regions:
                    regions.append(region_id)
                if len(regions) >= max_regions:
                    return regions
        return regions

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                {"id": n, **{k: v for k, v in data.items()}}
                for n, data in sorted(self.graph.nodes(data=True))
            ],
            "edges": [
                {
                    "head": u,
                    "tail": v,
                    "relation_type": data.get("relation_type", key),
                    "source_region_id": data.get("source_region_id"),
                }
                for u, v, key, data in self.graph.edges(keys=True, data=True)
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> KnowledgeGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = nx.MultiDiGraph()
        for node in payload["nodes"]:
            node_id = node.pop("id")
            graph.add_node(node_id, **node)
        for edge in payload["edges"]:
            graph.add_edge(
                edge["head"],
                edge["tail"],
                key=edge["relation_type"],
                relation_type=edge["relation_type"],
                source_region_id=edge.get("source_region_id"),
            )
        return KnowledgeGraph(graph)

    def describe_subgraph(self, nodes: Iterable[str], max_edges: int = 20) -> str:
        """Render a subgraph as text for the generator's prompt.

        The LLM reasons over this alongside the retrieved passages, which is
        how structured knowledge actually reaches the answer.
        """
        node_set = set(nodes)
        lines: list[str] = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_set and v in node_set:
                head = self.graph.nodes[u].get("name", u)
                tail = self.graph.nodes[v].get("name", v)
                lines.append(f"- {head} --{data.get('relation_type','related')}--> {tail}")
            if len(lines) >= max_edges:
                break
        return "\n".join(lines) if lines else "(no graph relations found)"


def build_kg(extraction: Extraction) -> KnowledgeGraph:
    return KnowledgeGraph.from_extraction(extraction)
