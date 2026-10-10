"""Extract text, tables, and figure images from the three economics PDFs.

Low-trust pages (Appendix C) are written aside and never indexed.
User-supplied chat values are not part of this stage.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import fitz

from mmrag.config import CHUNKS_PATH, DOCS, EXCLUDED_PATH, FIGURES_DIR, ROOT, data_dir

LABEL_RE = re.compile(
    r"(Figure|Table)\s+((?:[12]|[A-C])-\d+)(\s*\(corrected\))?",
    re.IGNORECASE,
)
PAIR_RE = re.compile(
    r"Figures?\s+([A-C]-\d+)\s*(?:and|/)\s*([A-C]-\d+)",
    re.IGNORECASE,
)
INJECTION_RE = re.compile(
    r"(ignore (all )?previous instructions|instruction to (an? )?ai|you are now|system prompt)",
    re.IGNORECASE,
)


def normalize_label(kind: str, number: str, corrected: str | None) -> str:
    kind_n = "Figure" if kind.lower().startswith("fig") else "Table"
    corr = " (corrected)" if corrected else ""
    return f"{kind_n} {number.upper()}{corr}"


def find_labels(text: str) -> list[str]:
    out = []
    for kind, number, corrected in LABEL_RE.findall(text or ""):
        label = normalize_label(kind, number, corrected)
        if label not in out:
            out.append(label)
    return out


def dehyphenate(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_injection(text: str) -> str:
    return INJECTION_RE.sub("[instruction inside a document removed]", text)


def page_is_low_trust(text: str, labels: list[str]) -> bool:
    """True only for Appendix C itself, not for earlier pages that mention it."""
    if any(lab.startswith("Figure C-") or lab.startswith("Table C-") for lab in labels):
        return True
    return bool(re.search(r"APPENDIX\s+C\s*:", text))


def sibling_map(pages_text: list[str]) -> dict[str, list[str]]:
    pairs: dict[str, set[str]] = {}
    for text in pages_text:
        for a, b in PAIR_RE.findall(text):
            la, lb = f"Figure {a.upper()}", f"Figure {b.upper()}"
            pairs.setdefault(la, set()).add(lb)
            pairs.setdefault(lb, set()).add(la)
    return {k: sorted(v) for k, v in pairs.items()}


def table_to_text(rows: list[list]) -> str:
    cleaned = []
    for row in rows:
        cells = []
        for cell in row:
            cell = "" if cell is None else str(cell)
            cell = re.sub(r"\s+", " ", cell).strip()
            cells.append(cell)
        if any(cells):
            cleaned.append(" | ".join(cells))
    return "\n".join(cleaned)


def caption_blocks(page: fitz.Page) -> list[dict]:
    found = []
    data = page.get_text("dict")
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        raw = " ".join(s["text"] for ln in block["lines"] for s in ln["spans"])
        raw = re.sub(r"\s+", " ", raw).strip()
        labels = find_labels(raw)
        if not labels:
            continue
        # A real caption starts with its label. Mentions inside a sentence are not captions.
        if not re.match(r"(Figure|Table)\s+", raw, re.IGNORECASE):
            continue
        if len(raw) > 240:
            continue
        found.append({"bbox": fitz.Rect(block["bbox"]), "text": raw, "labels": labels})
    return found


def drawing_clusters(page: fitz.Page) -> list[fitz.Rect]:
    rects = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect.width > 500 or rect.height > 650:
            continue
        if rect.width < 10 and rect.height < 10:
            continue
        if rect.y0 < 0 or rect.y1 > page.rect.height + 5:
            continue
        rects.append(rect)
    if not rects:
        return []
    rects.sort(key=lambda r: r.y0)
    clusters = [rects[0]]
    for rect in rects[1:]:
        if rect.y0 - clusters[-1].y1 > 36:
            clusters.append(rect)
        else:
            clusters[-1] = clusters[-1] | rect
    return [c for c in clusters if c.height > 30 and c.width > 40]


def _grow_vector_figure(page: fitz.Page, box: fitz.Rect) -> fitz.Rect:
    """Axis titles on textbook plots are text, not drawings. Pull those into the crop."""
    grown = fitz.Rect(box)
    search = fitz.Rect(30, box.y0 - 12, page.rect.width - 24, box.y1 + 42)
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        raw = " ".join(span["text"] for line in block["lines"] for span in line["spans"])
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > 160:
            continue
        rect = fitz.Rect(block["bbox"])
        if rect.intersects(search):
            grown |= rect
    grown = fitz.Rect(grown.x0 - 8, grown.y0 - 6, grown.x1 + 8, grown.y1 + 8)
    return grown & page.rect


def assign_region(caption: fitz.Rect, page: fitz.Page, images: list[fitz.Rect], clusters: list[fitz.Rect]) -> fitz.Rect:
    candidates = images + clusters
    if candidates:
        best = min(candidates, key=lambda r: min(abs(r.y0 - caption.y1), abs(r.y1 - caption.y0)))
        if min(abs(best.y0 - caption.y1), abs(best.y1 - caption.y0)) < 90:
            box = fitz.Rect(best.x0 - 4, best.y0 - 4, best.x1 + 4, best.y1 + 4) & page.rect
            # Embedded chart images already contain their axes. Vector plots do not.
            from_image = any(abs(best.y0 - img.y0) < 2 and abs(best.x0 - img.x0) < 2 for img in images)
            if from_image and box.width > 280 and box.height > 160:
                return box
            return _grow_vector_figure(page, box)
    y0 = min(caption.y1 + 2, page.rect.height - 40)
    y1 = min(y0 + 230, page.rect.height - 36)
    return fitz.Rect(48, y0, page.rect.width - 48, y1)


def render_clip(page: fitz.Page, box: fitz.Rect, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    clip = box & page.rect
    if clip.width < 20 or clip.height < 20:
        clip = page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=clip, alpha=False)
    pix.save(dest.as_posix())


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def revision_notes(text: str) -> list[str]:
    notes = []
    for match in re.finditer(
        r"revised from\s+([0-9.]+%?)\s*\(([^)]+)\)\s*to\s+([0-9.]+%?)",
        text,
        re.IGNORECASE,
    ):
        old, source, new = match.group(1), match.group(2), match.group(3)
        notes.append(
            f"REVISION: value changed from {old} ({source}) to {new}. "
            f"The revised value supersedes the earlier one."
        )
    if re.search(r"supersedes", text, re.IGNORECASE) and re.search(r"corrected", text, re.IGNORECASE):
        notes.append(
            "ERRATUM: a corrected figure supersedes the earlier printing of the same figure. "
            "Cite the corrected figure when a value differs."
        )
    return notes


def chunk_page_text(text: str, limit: int = 1800) -> list[str]:
    text = strip_injection(dehyphenate(text))
    # Drop running headers that repeat on every page.
    text = re.sub(r"\b(DOC[123]\s*\||Page\s+\d+)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + 400:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - 200
    return [p for p in parts if p]


def ingest() -> list[dict]:
    root = data_dir()
    if FIGURES_DIR.exists():
        shutil.rmtree(FIGURES_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_page_text: list[str] = []

    for spec in DOCS:
        path = root / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}. Place the competition PDFs next to Questions.csv.")
        pdf = fitz.open(path)
        for page in pdf:
            all_page_text.append(page.get_text("text"))
        pdf.close()

    siblings = sibling_map(all_page_text)
    chunks: list[dict] = []

    for spec in DOCS:
        pdf = fitz.open(root / spec["file"])
        for index, page in enumerate(pdf):
            page_no = index + 1
            raw = page.get_text("text")
            labels_on_page = find_labels(raw)
            low = page_is_low_trust(raw, labels_on_page)
            trust = "low" if low else spec["trust_level"]
            captions = caption_blocks(page)
            images = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
            clusters = drawing_clusters(page)
            used_boxes: list[fitz.Rect] = []

            figure_labels_rendered = set()
            for cap in captions:
                for label in cap["labels"]:
                    if not label.startswith("Figure"):
                        continue
                    if label in figure_labels_rendered:
                        continue
                    region = assign_region(cap["bbox"], page, images, clusters)
                    # Avoid handing the same image to two captions.
                    if any(region.intersects(prev) and abs(region.y0 - prev.y0) < 15 for prev in used_boxes):
                        continue
                    used_boxes.append(region)
                    image_path = FIGURES_DIR / f"{spec['doc_id']}_p{page_no}_{slug(label)}.png"
                    render_clip(page, region, image_path)
                    figure_labels_rendered.add(label)
                    text = (
                        f"{label}. Caption: {cap['text']}. "
                        f"This image is the only source for values of {label}. "
                        f"Do not copy numbers from a near-identical chart."
                    )
                    sibs = siblings.get(re.sub(r"\s*\(corrected\)", "", label), [])
                    if sibs:
                        text += " Near-identical figures that must stay separate: " + ", ".join(sibs) + "."
                    chunks.append(
                        {
                            "chunk_id": f"{spec['doc_id']}_p{page_no}_{slug(label)}",
                            "doc_id": spec["doc_id"],
                            "page": page_no,
                            "trust_level": trust,
                            "type": "figure",
                            "label": label,
                            "sibling_labels": sibs,
                            "text": strip_injection(text),
                            "image_path": str(image_path.relative_to(ROOT)),
                        }
                    )

            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []
            for t_i, table in enumerate(tables):
                body = table_to_text(table.extract() or [])
                if len(body) < 12:
                    continue
                nearby = find_labels(body)
                # Caption above the table.
                label = ""
                for cap in captions:
                    if cap["bbox"].y1 <= table.bbox[1] + 8:
                        for lab in cap["labels"]:
                            if lab.startswith("Table"):
                                label = lab
                if not label and nearby:
                    label = next((lab for lab in nearby if lab.startswith("Table")), nearby[0])
                header = f"{label}\n" if label else ""
                chunks.append(
                    {
                        "chunk_id": f"{spec['doc_id']}_p{page_no}_table{t_i}_{slug(label) or 'raw'}",
                        "doc_id": spec["doc_id"],
                        "page": page_no,
                        "trust_level": trust,
                        "type": "table",
                        "label": label,
                        "sibling_labels": [],
                        "text": strip_injection(header + body),
                        "image_path": "",
                    }
                )

            for note in revision_notes(raw):
                chunks.append(
                    {
                        "chunk_id": f"{spec['doc_id']}_p{page_no}_revision_{len(chunks)}",
                        "doc_id": spec["doc_id"],
                        "page": page_no,
                        "trust_level": trust,
                        "type": "revision",
                        "label": "",
                        "sibling_labels": [],
                        "text": strip_injection(note + " Context: " + dehyphenate(raw)[:900]),
                        "image_path": "",
                    }
                )

            for part_i, part in enumerate(chunk_page_text(raw)):
                if len(part) < 80:
                    continue
                chunks.append(
                    {
                        "chunk_id": f"{spec['doc_id']}_p{page_no}_text{part_i}",
                        "doc_id": spec["doc_id"],
                        "page": page_no,
                        "trust_level": trust,
                        "type": "text",
                        "label": "",
                        "sibling_labels": [],
                        "text": part,
                        "image_path": "",
                    }
                )
        pdf.close()

    by_label = {c["label"]: c for c in chunks if c.get("label")}
    original = by_label.get("Figure B-5")
    corrected = by_label.get("Figure B-5 (corrected)")
    if original and corrected:
        original["sibling_labels"] = sorted(set(original["sibling_labels"] + ["Figure B-5 (corrected)"]))
        corrected["sibling_labels"] = sorted(set(corrected["sibling_labels"] + ["Figure B-5"]))
        original["text"] += " The corrected figure supersedes this one where the 2012 value differs."
        corrected["text"] += " This corrected figure supersedes the original Figure B-5 where the 2012 value differs."

    kept, dropped = [], []
    for chunk in chunks:
        if chunk["trust_level"] == "low":
            dropped.append(chunk)
        else:
            kept.append(chunk)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as handle:
        for chunk in kept:
            handle.write(json.dumps(chunk) + "\n")
    with EXCLUDED_PATH.open("w") as handle:
        for chunk in dropped:
            handle.write(json.dumps(chunk) + "\n")

    print(f"indexed chunks: {len(kept)}")
    print(f"excluded low-trust chunks: {len(dropped)}")
    print(f"figure images: {len(list(FIGURES_DIR.glob('*.png')))}")
    print(f"wrote {CHUNKS_PATH}")
    return kept


if __name__ == "__main__":
    ingest()
