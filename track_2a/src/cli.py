import argparse
import json
from pathlib import Path

from src.analyzer import analyze_case


def load_cases(path: Path) -> list[dict]:
    cases = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def write_predictions(
    path: Path,
    predictions: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="HackApertus Track 2A OST solution."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if input_path == output_path:
        parser.error(
            "Input and output must be different files."
        )

    cases = load_cases(input_path)

    predictions = []

    print("=" * 80)
    print("HACKAPERTUS TRACK 2A - OST")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Cases:  {len(cases)}")
    print(f"Output: {output_path}")
    print()

    for position, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{position}/{len(cases)}] "
            f"{case['id']}"
        )

        prediction = analyze_case(
            case,
            input_path,
        )

        predictions.append(prediction)

        print(
            f"  label={prediction['label']} "
            f"({prediction['label_name']}) "
            f"input_tokens="
            f"{prediction['metrics']['input_tokens']} "
            f"time_ms="
            f"{prediction['metrics']['inference_time_ms']:.1f}"
        )

    write_predictions(
        output_path,
        predictions,
    )

    print()
    print(
        f"Wrote {len(predictions)} predictions "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()