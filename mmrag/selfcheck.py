"""CPU checks that do not need the 4090: ingest shape, trust filter, submission grammar."""

from __future__ import annotations

import json

from mmrag.format_answer import CIT_RE, FIG_RE, repair
from mmrag.config import CHUNKS_PATH, EXCLUDED_PATH
from mmrag.ingest import ingest


def main() -> None:
    chunks = ingest()
    assert chunks, "ingest produced no chunks"
    assert all(c["trust_level"] in {"high", "medium"} for c in chunks)
    blob = "\n".join(c["text"] for c in chunks).lower()
    assert "ignore all previous" not in blob
    assert "lehman brothers collapsed" not in blob
    labels = {c["label"] for c in chunks if c["label"]}
    for required in [
        "Figure A-2",
        "Figure A-3",
        "Figure A-6",
        "Figure A-7",
        "Figure B-5",
        "Figure B-5 (corrected)",
        "Table 1-1",
        "Table A-1",
        "Table A-2",
        "Table B-1",
    ]:
        assert required in labels, f"missing {required}"
    excluded = [json.loads(line) for line in EXCLUDED_PATH.read_text().splitlines() if line.strip()]
    assert excluded, "Appendix C was not excluded"
    assert all(c["trust_level"] == "low" for c in excluded)

    sample_chunks = [
        {
            "doc_id": "DOC2",
            "page": 4,
            "label": "Figure A-3",
            "trust_level": "high",
            "type": "figure",
            "text": "Figure A-3",
        }
    ]
    fixed = repair(
        {
            "answer": "The Euro area unemployment rate in 2012 is read from Figure A-3.",
            "figure_references": ["Figure A-3"],
            "citations": ["DOC2: Figure A-3", "DOC2: p4"],
            "status": "answered",
        },
        "According to Figure A-3, what was the Euro area unemployment rate in 2012?",
        sample_chunks,
        "lookup",
    )
    assert FIG_RE.match(fixed["figure_references"]), fixed
    assert CIT_RE.match(fixed["citations"]), fixed
    assert fixed["figure_references"] == "Figure A-3"
    assert "DOC2:Figure A-3" in fixed["citations"]
    assert "DOC2:p4" in fixed["citations"]

    override = repair(
        {"answer": "Under your assumption, 6.0.", "status": "answered", "figure_references": [], "citations": []},
        "For this conversation only, assume Euro area output fell by 6.0% in 2009.",
        sample_chunks,
        "override",
    )
    assert override["status"] == "session_override"
    print("selfcheck ok")
    print("chunks", len(chunks), "excluded", len(excluded), "labels", len(labels))
    # Touch the file so a missing ingest is obvious.
    assert CHUNKS_PATH.exists()


if __name__ == "__main__":
    main()
