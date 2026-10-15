import json
import math
import re
from collections import Counter
from pathlib import Path

from src.test_segmentation import segment_reference_with_fallback


DATASET = Path("data/v1.1.jsonl")

TOP_K = 3


# Common German / French / Italian function words.
# These are intentionally small and conservative.
STOPWORDS = {
    # German
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "einem", "einen",
    "und", "oder", "ist", "sind", "wird", "werden",
    "von", "für", "mit", "auf", "im", "in", "an",
    "zu", "des", "als", "nach", "auch", "nicht",

    # French
    "le", "la", "les", "un", "une", "des",
    "du", "de", "et", "ou", "est", "sont",
    "pour", "avec", "dans", "sur", "en", "au",
    "aux", "ce", "cette", "ces", "qui", "que",
    "pas", "plus", "par",

    # Italian
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una", "di", "del", "della",
    "e", "o", "è", "sono", "per", "con",
    "in", "su", "al", "alla", "che", "non",
    "più", "da",
}


def normalize(text):
    text = text.lower()

    # Keep letters/numbers and spaces.
    text = re.sub(r"[^\w\s%€.,:/+-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    normalized = normalize(text)

    tokens = re.findall(
        r"\b[\wÀ-ÿ]+\b",
        normalized,
        flags=re.UNICODE,
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) >= 2
    ]


def extract_anchors(text):
    """
    Extract high-value literal anchors that often survive
    across languages:

    - numbers
    - percentages
    - dates
    - monetary values
    - article references
    """

    normalized = normalize(text)

    anchors = set()

    # Numbers, including decimals and percentages.
    for match in re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b",
        normalized,
    ):
        anchors.add(match)

    # Dates such as 17 March 2023 / 17.03.2023 / 17/03/2023.
    for match in re.findall(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        normalized,
    ):
        anchors.add(match)

    # Article references.
    for match in re.findall(
        r"\bart\.?\s*\d+[a-z]?(?:\s+abs\.?\s*\d+)?",
        normalized,
    ):
        anchors.add(match)

    return anchors


def cosine_similarity(counter_a, counter_b):
    if not counter_a or not counter_b:
        return 0.0

    common = set(counter_a) & set(counter_b)

    numerator = sum(
        counter_a[token] * counter_b[token]
        for token in common
    )

    norm_a = math.sqrt(
        sum(value * value for value in counter_a.values())
    )

    norm_b = math.sqrt(
        sum(value * value for value in counter_b.values())
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def character_ngrams(text, n=3):
    normalized = normalize(text)

    padded = f"  {normalized}  "

    return Counter(
        padded[index:index + n]
        for index in range(len(padded) - n + 1)
    )


def token_similarity(claim_tokens, segment_tokens):
    claim_counter = Counter(claim_tokens)
    segment_counter = Counter(segment_tokens)

    return cosine_similarity(claim_counter, segment_counter)


def character_similarity(claim, segment):
    claim_ngrams = character_ngrams(claim)
    segment_ngrams = character_ngrams(segment)

    return cosine_similarity(claim_ngrams, segment_ngrams)


def anchor_score(claim, segment):
    claim_anchors = extract_anchors(claim)
    segment_anchors = extract_anchors(segment)

    if not claim_anchors:
        return 0.0

    matches = claim_anchors & segment_anchors

    return len(matches) / len(claim_anchors)


def heading_score(claim, heading):
    if not heading:
        return 0.0

    claim_tokens = set(tokenize(claim))
    heading_tokens = set(tokenize(heading))

    if not claim_tokens or not heading_tokens:
        return 0.0

    overlap = claim_tokens & heading_tokens

    return len(overlap) / len(claim_tokens)


def rank_evidence(claim, reference):
    segments = segment_reference_with_fallback(reference)

    ranked = []

    for index, segment in enumerate(segments):
        text = segment["text"]
        heading = segment.get("heading")

        tokens = tokenize(text)

        token_score = token_similarity(
            tokenize(claim),
            tokens,
        )

        char_score = character_similarity(
            claim,
            text,
        )

        anchors = anchor_score(
            claim,
            text,
        )

        heading_match = heading_score(
            claim,
            heading,
        )

        # Hybrid score.
        #
        # Character similarity is deliberately given the largest
        # weight because it can retain useful signal across
        # languages without requiring translation.
        #
        # Anchors receive a strong weight because exact numbers,
        # percentages, dates and article references are highly
        # discriminative in voting-booklet text.
        score = (
            0.40 * char_score
            + 0.25 * token_score
            + 0.25 * anchors
            + 0.10 * heading_match
        )

        ranked.append(
            {
                "id": index,
                "heading": heading,
                "text": text,
                "score": score,
                "char_score": char_score,
                "token_score": token_score,
                "anchor_score": anchors,
                "heading_score": heading_match,
                "chars": len(text),
            }
        )

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


def load_dataset():
    rows = []

    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():
    rows = load_dataset()

    indices = [
        14,
        495,
        1032,
        292,
        661,
        172,
        1468,
        827,
        1340,
    ]

    print("=" * 80)
    print("HYBRID EVIDENCE RETRIEVAL — V1.1")
    print("=" * 80)

    for index in indices:
        row = rows[index]

        print("\n" + "=" * 80)
        print(
            f"INDEX {index} | "
            f"GOLD={row['entailment_label']} | "
            f"{row['claim_language']} -> {row['reference_language']}"
        )

        print("\nCLAIM:")
        print(row["claim"])

        ranked = rank_evidence(
            row["claim"],
            row["reference_string"],
        )

        selected = ranked[:TOP_K]

        print("\nTOP CANDIDATES:")

        for item in selected:
            print(
                f"\nRANK {item['rank']} "
                f"| SEGMENT {item['id']} "
                f"| SCORE {item['score']:.4f} "
                f"| CHARS {item['chars']}"
            )

            print(
                f"  char={item['char_score']:.4f} "
                f"token={item['token_score']:.4f} "
                f"anchor={item['anchor_score']:.4f} "
                f"heading={item['heading_score']:.4f}"
            )

            print(f"  HEADING: {item['heading']}")
            print(f"  TEXT: {item['text'][:700]}")

        selected_chars = sum(
            item["chars"]
            for item in selected
        )

        reference_chars = len(row["reference_string"])

        print(
            f"\nSELECTED CHARS: "
            f"{selected_chars} / {reference_chars} "
            f"({selected_chars / reference_chars:.1%})"
        )


if __name__ == "__main__":
    main()