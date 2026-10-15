from pathlib import Path
import re

import pypdf

from src.hybrid_evidence_ranker import rank_evidence


def _normalized_with_offsets(source: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining offsets into the original text."""
    chars = []
    offsets = []
    index = 0

    while index < len(source):
        if source[index].isspace():
            start = index
            while index < len(source) and source[index].isspace():
                index += 1

            if chars and chars[-1] != " ":
                chars.append(" ")
                offsets.append(start)
        else:
            chars.append(source[index])
            offsets.append(index)
            index += 1

    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()

    return "".join(chars), offsets


def _find_verbatim_span(source: str, passage: str) -> str | None:
    """Recover an exact source substring using whitespace-normalized matching."""
    normalized_source, offsets = _normalized_with_offsets(source)
    normalized_passage = re.sub(r"\s+", " ", passage).strip()

    if not normalized_passage:
        return None

    start = normalized_source.find(normalized_passage)

    if start < 0:
        return None

    end = start + len(normalized_passage)
    original_start = offsets[start]
    original_end = offsets[end - 1] + 1

    result = source[original_start:original_end]

    # Confirm that the recovered source span represents the ranked passage.
    if re.sub(r"\s+", " ", result).strip() != normalized_passage:
        return None

    return result


def rank_pdf_evidence(claim: str, pdf_path: Path) -> list[dict]:
    """Rank PDF passages globally while retaining page and source-text provenance."""
    reader = pypdf.PdfReader(str(pdf_path))
    ranked = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue

        for item in rank_evidence(claim, page_text):
            text = item.get("text", "")

            if not text.strip():
                continue

            verbatim_text = _find_verbatim_span(page_text, text)

            # Never label normalized or unverified text as verbatim evidence.
            if verbatim_text is None:
                continue

            ranked.append({
                **item,
                "page": page_number,
                "verbatim_text": verbatim_text,
            })

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["anchor_score"],
            item["char_score"],
            item["token_score"],
        ),
        reverse=True,
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked
