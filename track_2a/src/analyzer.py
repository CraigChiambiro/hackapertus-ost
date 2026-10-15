import time
from pathlib import Path

import pypdf

from src.hybrid_evidence_ranker import rank_evidence
from src.page_aware_evidence import rank_pdf_evidence
from src.nli import classify_claim


LABEL_NAMES = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

MAX_EVIDENCE = 5
MAX_EVIDENCE_CHARS = 5000
NLI_TOP_K = 3


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all non-empty pages of a PDF."""
    reader = pypdf.PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n\n".join(pages)


def resolve_booklet_path(
    input_path: Path,
    relative_path: str,
) -> Path:
    """Resolve a booklet path relative to the cases file."""
    return (input_path.parent / relative_path).resolve()


def build_evidence(
    ranked: list[dict],
) -> list[dict]:
    """Format evidence with page provenance and source text."""
    evidence = []

    for item in ranked:
        if len(evidence) >= MAX_EVIDENCE:
            break

        text = item.get("verbatim_text", item["text"]).strip()

        if not text:
            continue

        evidence.append({
            "page": item.get("page"),
            "text": text[:MAX_EVIDENCE_CHARS],
        })

    return evidence


def build_nli_context(
    ranked: list[dict],
) -> str:
    """Build the compact context from the same ranked passages."""
    selected = ranked[:NLI_TOP_K]

    return "\n\n".join(
        item["text"]
        for item in selected
        if item["text"].strip()
    )

def load_reference(
    case: dict,
    input_path: Path,
) -> str:
    """Load either a direct reference or a booklet PDF."""
    case_id = case["id"]

    if "reference" in case:
        return case["reference"]["text"]

    if "booklet" in case:
        booklet_path = resolve_booklet_path(
            input_path,
            case["booklet"]["path"],
        )

        if not booklet_path.exists():
            raise FileNotFoundError(
                f"Booklet not found for {case_id}: {booklet_path}"
            )

        return extract_pdf_text(booklet_path)

    raise ValueError(
        f"Case {case_id} contains neither "
        "'reference' nor 'booklet'."
    )


def parse_prediction(raw_output: str) -> int:
    """Parse a valid NLI label without silently guessing."""
    import re

    output = raw_output.strip()

    # Accept a bare label, optionally surrounded by whitespace or quotes.
    if re.fullmatch(r"""["']?[012]["']?""", output):
        return int(output.strip("\"'"))

    # Accept explicit forms such as "Label: 2" or "Prediction: 0".
    match = re.fullmatch(
        r"""(?:label|prediction)\s*:\s*([012])""",
        output,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Invalid NLI model output: {raw_output!r}"
    )


def analyze_case(
    case: dict,
    input_path: Path,
) -> dict:
    """
    Run the complete OST analysis pipeline for one case.

    Pipeline:
        input case
        -> document/reference loading
        -> evidence retrieval
        -> compact NLI context
        -> Apertus classification
        -> evaluator-compatible result
    """
    case_id = case["id"]
    claim = case["claim"]["text"]

    start = time.perf_counter()

    reference = load_reference(
        case,
        input_path,
    )

    if not reference.strip():
        raise ValueError(
            f"Case {case_id} produced an empty reference."
        )

    if "booklet" in case and "reference" not in case:
        booklet_path = resolve_booklet_path(
            input_path,
            case["booklet"]["path"],
        )
        ranked = rank_pdf_evidence(claim, booklet_path)
    else:
        ranked = rank_evidence(claim, reference)
        ranked = [
            {**item, "page": None}
            for item in ranked
        ]

    evidence = build_evidence(ranked)
    nli_context = build_nli_context(ranked)

    if not nli_context.strip():
        nli_context = reference[:15000]

    result = classify_claim(
        claim,
        nli_context,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    prediction = parse_prediction(
        result["raw_output"]
    )

    usage = result.get("usage") or {}

    input_tokens = usage.get(
        "prompt_tokens",
        0,
    )

    output_tokens = usage.get(
        "completion_tokens",
        0,
    )

    return {
        "id": case_id,
        "label": prediction,
        "label_name": LABEL_NAMES[prediction],
        "raw_output": result["raw_output"],
        "evidence": evidence,
        "metrics": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time_ms": round(
                elapsed_ms,
                3,
            ),
        },
    }
