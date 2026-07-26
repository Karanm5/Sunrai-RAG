"""Generate PubLayNet-shaped test fixtures: page renders + COCO annotations.

This exists because the real dataset is not reachable from every environment,
and because a fixture whose ground truth is *known exactly* is better for
verifying the ingestion path than a real page whose correct OCR output we
would have to eyeball.

The renders imitate what PubLayNet actually contains: a rendered PMC page
image plus a COCO-style annotation listing bounding boxes over five region
classes. Critically, the annotation carries NO text -- exactly like the real
thing -- so OCR is genuinely exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAGE_W, PAGE_H = 1000, 1300
CATEGORY = {"text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}


def _font(size: int):
    """Best available font; falls back to PIL's bitmap default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_page(doc_id: str, page_no: int, spec: list[dict], out_dir: Path) -> dict:
    """Render one page and write the matching COCO annotation.

    `spec` entries: {"type": ..., "text": ..., "y": ...}
    Returns the ground truth so a test can assert what OCR *should* recover.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(image)

    annotations: list[dict] = []
    truth: list[dict] = []
    margin = 60

    for item in spec:
        rtype, text, y = item["type"], item["text"], item["y"]
        size = 34 if rtype == "title" else 21
        font = _font(size)
        max_w = PAGE_W - 2 * margin

        if rtype == "figure":
            # A figure: bordered box with a bar chart and a caption beneath.
            h = item.get("h", 240)
            draw.rectangle([margin, y, PAGE_W - margin, y + h], outline="black", width=2)
            bars = item.get("bars", [0.5, 0.7, 0.9])
            bw, base = 90, y + h - 40
            for i, frac in enumerate(bars):
                x0 = margin + 90 + i * (bw + 55)
                draw.rectangle(
                    [x0, base - int(frac * (h - 90)), x0 + bw, base], fill="#4a5568"
                )
            cap_font = _font(19)
            draw.text((margin + 8, y + h + 8), text, fill="black", font=cap_font)
            box_h = h + 34
            annotations.append(
                {"bbox": [margin, y, PAGE_W - 2 * margin, box_h],
                 "category_id": CATEGORY["figure"], "area": (PAGE_W - 2 * margin) * box_h}
            )
            truth.append({"type": "figure", "expected_text": text})
            continue

        if rtype == "table":
            rows = item.get("rows", [])
            row_h, tbl_h = 34, 34 * (len(rows) + 1)
            draw.rectangle([margin, y, PAGE_W - margin, y + tbl_h], outline="black", width=2)
            tf = _font(19)
            for r, row in enumerate(rows):
                ry = y + 8 + r * row_h
                draw.text((margin + 14, ry), row, fill="black", font=tf)
                if r:
                    draw.line([margin, ry - 6, PAGE_W - margin, ry - 6], fill="#999")
            draw.text((margin + 8, y + tbl_h + 8), text, fill="black", font=_font(19))
            box_h = tbl_h + 34
            annotations.append(
                {"bbox": [margin, y, PAGE_W - 2 * margin, box_h],
                 "category_id": CATEGORY["table"], "area": (PAGE_W - 2 * margin) * box_h}
            )
            truth.append({"type": "table", "expected_text": text})
            continue

        lines = _wrap(draw, text, font, max_w)
        line_h = size + 9
        for i, line in enumerate(lines):
            draw.text((margin, y + i * line_h), line, fill="black", font=font)
        box_h = len(lines) * line_h + 8
        annotations.append(
            {"bbox": [margin, y - 4, PAGE_W - 2 * margin, box_h],
             "category_id": CATEGORY[rtype], "area": (PAGE_W - 2 * margin) * box_h}
        )
        truth.append({"type": rtype, "expected_text": text})

    key = f"{doc_id}_{page_no:05d}"
    image.save(out_dir / f"{key}.png")
    # Note the shape: boxes and category ids only. No text -- as in the real dataset.
    (out_dir / f"{key}.json").write_text(
        json.dumps({"annotations": annotations}, indent=2), encoding="utf-8"
    )
    return {"key": key, "truth": truth}


def build_fixture_corpus(out_dir: Path, n_docs: int = 3) -> list[dict]:
    """A small multi-document corpus with the visual-only-answer property."""
    methods = ["random forest", "convolutional neural network", "support vector machine"]
    manifest = []
    for d in range(n_docs):
        method, acc = methods[d], 80 + d * 4
        manifest.append(render_page(f"PMC{5000000 + d}", 1, [
            {"type": "title", "y": 70,
             "text": f"Predictive modelling of clinical outcomes using {method}"},
            {"type": "text", "y": 160,
             "text": ("Abstract. We evaluate machine learning approaches for predicting "
                      "patient outcomes in a prospective cohort. Models were trained on "
                      "routinely collected clinical variables and validated using "
                      "stratified cross validation across five folds.")},
            {"type": "text", "y": 330,
             "text": (f"Methods. A {method} was trained on the clinical cohort. "
                      "Performance was assessed using accuracy and area under the curve. "
                      "Complete numerical results are reported in Figure 1 below.")},
            # The accuracy VALUE appears only in this caption -- never in body text.
            {"type": "figure", "y": 500, "h": 230, "bars": [0.55, 0.72, 0.93],
             "text": f"Figure 1: accuracy of {acc}.5 percent achieved by the {method}"},
            {"type": "text", "y": 810,
             "text": ("Discussion. These findings suggest that routinely collected "
                      "variables carry substantial predictive signal for this outcome.")},
        ], out_dir))
        manifest.append(render_page(f"PMC{5000000 + d}", 2, [
            {"type": "text", "y": 70,
             "text": ("Data collection. Records were gathered prospectively across three "
                      "hospital sites over a twelve month period. Preprocessing followed "
                      "standard normalisation and missing values were imputed.")},
            {"type": "table", "y": 260,
             "rows": ["Cohort        Patients    Events",
                      "Training         1840        212",
                      "Validation        610         74"],
             "text": "Table 1: cohort characteristics for training and validation"},
            {"type": "list", "y": 470,
             "text": ("1. Inclusion required at least one recorded visit. "
                      "2. Patients under eighteen were excluded. "
                      "3. Incomplete records were removed before analysis.")},
        ], out_dir))
    return manifest


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/pages")
    manifest = build_fixture_corpus(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {len(manifest)} pages to {out}")
