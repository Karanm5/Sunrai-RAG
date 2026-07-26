"""Config validation and determinism controls."""
import pytest
import yaml

from sunrai_rag.config import Config, ConfigError, load_config, set_global_seeds


def _write(tmp_path, payload):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(payload))
    return p


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.seed == 42 and cfg.retrieval.top_k > 0


def test_loads_and_overrides_from_yaml(tmp_path):
    cfg = load_config(_write(tmp_path, {"seed": 7, "retrieval": {"top_k": 3}}))
    assert cfg.seed == 7 and cfg.retrieval.top_k == 3


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    """A silently-ignored typo is a reproducibility bug."""
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(_write(tmp_path, {"retrieval": {"top_kk": 3}}))


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(_write(tmp_path, {"nonsense": {"a": 1}}))


def test_candidate_k_must_not_be_below_top_k(tmp_path):
    with pytest.raises(ConfigError, match="candidate_k"):
        load_config(_write(tmp_path, {"retrieval": {"top_k": 10, "candidate_k": 5}}))


def test_primary_k_must_be_reported(tmp_path):
    with pytest.raises(ConfigError, match="primary_k"):
        load_config(_write(tmp_path, {"eval": {"k_values": [1, 3], "primary_k": 5}}))


def test_nonzero_temperature_rejected_to_protect_determinism(tmp_path):
    with pytest.raises(ConfigError, match="reproducible"):
        load_config(_write(tmp_path, {"llm": {"temperature": 0.7}}))


@pytest.mark.parametrize("section,payload,match", [
    ("llm", {"backend": "wat"}, "llm.backend"),
    ("kg", {"extractor": "wat"}, "kg.extractor"),
    ("chunk", {"strategy": "wat"}, "chunk.strategy"),
])
def test_enumerated_options_validated(tmp_path, section, payload, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, {section: payload}))


def test_overlap_must_be_smaller_than_window(tmp_path):
    with pytest.raises(ConfigError, match="overlap"):
        load_config(_write(tmp_path, {"chunk": {"window_chars": 100, "overlap_chars": 100}}))


def test_top_level_must_be_a_mapping(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


def test_seeding_makes_random_reproducible():
    import random
    set_global_seeds(123); a = [random.random() for _ in range(5)]
    set_global_seeds(123); b = [random.random() for _ in range(5)]
    assert a == b


def test_seeding_makes_numpy_reproducible():
    import numpy as np
    set_global_seeds(99); a = np.random.rand(5).tolist()
    set_global_seeds(99); b = np.random.rand(5).tolist()
    assert a == b


def test_ensure_dirs_creates_all_paths(tmp_path):
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "d")
    cfg.paths.artifacts_dir = str(tmp_path / "a")
    cfg.paths.results_dir = str(tmp_path / "r")
    cfg.paths.cache_dir = str(tmp_path / "a" / "cache")
    cfg.ensure_dirs()
    assert all((tmp_path / n).exists() for n in ["d", "a", "r"])


def test_config_serialises_for_the_results_provenance_trail():
    assert "retrieval" in Config().to_dict()
