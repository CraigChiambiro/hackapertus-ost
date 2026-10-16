# ApertusVote
### Multilingual Natural Language Inference over Swiss Voting Booklets

ApertusVote is my submission for the OST challenge in Track 2A of the HackApertus Online Hackathon.

The goal is straightforward: given a claim and a Swiss official voting booklet, determine whether the booklet supports the claim, contradicts it, or does not provide enough information to decide.

The system uses Apertus-v1.5-8B through the CSCS inference API and retrieves relevant passages from reference text or extracted PDF content to support its prediction.

## What it does

- Classifies claims as entailment, neutral, or contradiction.
- Retrieves supporting evidence from reference material and voting-booklet PDFs.
- Preserves page references for PDF evidence when available.
- Records prediction labels, evidence, token usage, and inference timing.
- Runs in Docker so the execution environment is reproducible.

### Prediction labels

| Label | Meaning |
|---|---|
| `0` | Entailment — the reference supports the claim |
| `1` | Neutral — the reference does not establish whether the claim is true |
| `2` | Contradiction — the reference conflicts with the claim |

## Run the project

The Track 2A project is kept in `track_2a/`, as required by the challenge template.

```bash
cd track_2a
