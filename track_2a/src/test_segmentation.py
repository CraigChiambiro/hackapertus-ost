import json
import re
from pathlib import Path


DATASET = Path("data/v1.0.jsonl")


POSITION_MARKERS = {
    "OUI",
    "NON",
    "YES",
    "NO",
    "SI",
    "SÌ",
}


def is_likely_heading(line: str) -> bool:
    text = line.strip()

    if not text:
        return False

    if text.upper() in POSITION_MARKERS:
        return True

    if re.match(r"^\s*\d+[\.\)]\s+\S+", text):
        return True

    letters = [char for char in text if char.isalpha()]

    if len(letters) >= 3 and text == text.upper():
        words = text.split()

        if len(words) <= 12 and len(text) <= 120:
            return True

    return False


def expand_inline_position_markers(lines: list[str]) -> list[str]:
    expanded = []

    # These markers are often flattened into the same line as
    # surrounding booklet text. Split only uppercase OUI./NON.
    # markers followed by whitespace.
    pattern = re.compile(r"(?<!\w)(OUI|NON)\.(?=\s)")

    for line in lines:
        parts = pattern.split(line)

        if len(parts) == 1:
            expanded.append(line)
            continue

        buffer = ""

        for part in parts:
            if part in POSITION_MARKERS:
                if buffer.strip():
                    expanded.append(buffer.strip())

                expanded.append(part)
                buffer = ""
            else:
                buffer += part

        if buffer.strip():
            expanded.append(buffer.strip())

    return expanded


def segment_reference(reference: str) -> list[dict]:
    lines = [
        line.strip()
        for line in reference.splitlines()
        if line.strip()
    ]

    lines = expand_inline_position_markers(lines)

    if not lines:
        return []

    segments = []
    current_title = None
    current_text = []

    for line in lines:
        if is_likely_heading(line):
            if current_title is not None or current_text:
                segments.append(
                    {
                        "heading": current_title,
                        "text": " ".join(current_text).strip(),
                    }
                )

            current_title = line
            current_text = []

        else:
            current_text.append(line)

    if current_title is not None or current_text:
        segments.append(
            {
                "heading": current_title,
                "text": " ".join(current_text).strip(),
            }
        )

    return segments


def segment_reference_with_fallback(reference: str) -> list[dict]:
    """
    Segment a reference using the existing structural parser first.

    For long references where the structural parser produces too few
    segments, fall back to legal article boundaries.
    """

    primary_segments = segment_reference(reference)

    # Keep the existing parser when it already provides useful structure.
    if len(reference) < 10000 or len(primary_segments) >= 4:
        return primary_segments

    article_pattern = re.compile(
        r"(?<!\w)Art\.\s*\d+[a-z]?(?:\s+Abs\.)?"
    )

    matches = list(article_pattern.finditer(reference))

    # If there are not enough article markers, keep the original result.
    if len(matches) < 2:
        return primary_segments

    fallback_segments = []

    # Preserve text appearing before the first article.
    if matches[0].start() > 0:
        intro = reference[:matches[0].start()].strip()

        if intro:
            fallback_segments.append(
                {
                    "heading": "INTRODUCTION",
                    "text": intro,
                }
            )

    # Create one segment per legal article boundary.
    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(reference)

        text = reference[start:end].strip()

        if not text:
            continue

        fallback_segments.append(
            {
                "heading": match.group().strip(),
                "text": text,
            }
        )

    return fallback_segments


def load_rows():
    with DATASET.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    rows = load_rows()

    selected = [
        0,
        498,
        996,
    ]

    for index in selected:
        row = rows[index]

        print("=" * 80)
        print(
            f"INDEX: {index} | "
            f"LABEL: {row['entailment_label']} | "
            f"CLAIM: {row['claim_language']} | "
            f"REFERENCE: {row['reference_language']}"
        )
        print("=" * 80)

        segments = segment_reference(row["reference_string"])

        print(f"Segments detected: {len(segments)}")
        print()

        for number, segment in enumerate(segments, start=1):
            heading = segment["heading"] or "[NO HEADING]"

            print(f"[SEGMENT {number}]")
            print(f"HEADING: {heading}")
            print(f"TEXT: {segment['text'][:500]}")

            if len(segment["text"]) > 500:
                print("...")

            print()

    print("=" * 80)
    print("SEGMENTATION TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()