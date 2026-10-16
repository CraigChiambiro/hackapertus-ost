# ApertusVote — Multilingual Voting Claim Verification

**HackApertus Online Hackathon · Track 2A · OST**

I'm building ApertusVote to check claims against Swiss official voting booklets. The idea is simple: give the system a claim and reference material, and it predicts whether the reference supports the claim, contradicts it, or doesn't provide enough information to decide.

The project uses **Apertus-v1.5-8B** through the CSCS inference API. It combines evidence retrieval with model-based classification so that predictions can be accompanied by relevant passages from the source material.

## What it does

- Classifies claims as **entailment**, **neutral**, or **contradiction**.
- Retrieves relevant passages from reference text and voting-booklet PDFs.
- Preserves page references for PDF evidence when available.
- Returns prediction labels, evidence, and available token-usage and inference-time metrics.
- Runs through Docker using the project's `Makefile`.

## Prediction labels

| Label | Meaning |
|---|---|
| `0` | Entailment — the reference supports the claim. |
| `1` | Neutral — the reference does not establish whether the claim is true. |
| `2` | Contradiction — the reference conflicts with the claim. |

## Run it

From the repository root:

```bash
cd track_2a
make run
```

The project requires Docker and access to the CSCS Apertus inference API. Configure the environment variables documented in `track_2a/README.md` and the technical report. Keep API keys in a local `.env` file; never commit credentials.

## Testing

I verified that the Docker workflow can call Apertus and produce predictions. Development checks included:

- A single-example live inference test.
- A 15-example diagnostic run.
- A 27-example multilingual diagnostic run involving German, French, and Italian claims and references.
- PDF evidence retrieval and page-reference checks.

These tests demonstrate that the prototype runs, but they are **not official held-out benchmark results**. The diagnostic samples are small, classification errors remain possible, and official held-out evaluation is still pending.

## Project files

- [`track_2a/src/`](track_2a/src/) — implementation.
- [`track_2a/data/`](track_2a/data/) — input fixtures and project data.
- [`track_2a/Dockerfile`](track_2a/Dockerfile) — container setup.
- [`track_2a/Makefile`](track_2a/Makefile) — run workflow.
- [`track_2a/technical_report.md`](track_2a/technical_report.md) — technical details, testing, and limitations.

## Limitations

Evidence quality depends on PDF extraction and retrieval. Cross-language matching can fail, and a prediction is not necessarily correct just because the system returns evidence. Official held-out benchmark performance has not yet been established.

## Licensing

Please check the HackApertus terms and the applicable licences for third-party datasets and voting-booklet documents before redistributing those materials.

[HackApertus terms and conditions](https://hackapertus.ch/terms-and-conditions)
