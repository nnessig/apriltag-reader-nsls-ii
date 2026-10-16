---
tools-used:
  - ChatGPT
  - Claude Code
models-used:
  - gpt-5
  - gpt-5.1
  - gpt-5.2
  - gpt-5.4
  - claude-sonnet-5-5
providers:
  - OpenAI
  - Anthropic
scope:
  human-authored: >
    Project direction, hardware setup, camera calibration procedure,
    container configuration (Dockerfile, docker-compose.yml,
    entrypoint.sh), and final review of all merged changes.
  ai-assisted: >
    Source code (apriltag_runner.py, camera_calibration.py,
    check_container.py) and documentation (README.md,
    CONTRIBUTING.md, this file) — all with human review and
    validation.
last-updated: 2026-10-16
---

# AI Disclosure

This file describes how AI tools are used in this repository.
It is provided for transparency in accordance with the
[BNL AI Policy](https://intranet.bnl.gov/itd/ai-policy.php)
Transparency principle and
[BNL Generative AI Usage Guidelines](https://intranet.bnl.gov/itd/ai-policy.php).

## Disclosure levels

This repository uses the vocabulary defined by the
[W3C AI Content Disclosure Community Group](https://www.w3.org/community/ai-content-disclosure/)
(a community-group effort, not a ratified W3C standard).
Four levels are defined:

| Level | Meaning |
| --- | --- |
| `none` | No AI tools were used. |
| `ai-assisted` | AI contributed, but a human authored and reviewed. |
| `ai-generated` | AI generated the content; a human reviewed it. |
| `autonomous` | AI produced the content without human review. |

Per the community group's "absence = unknown" principle, files
without an explicit disclosure tag do not imply a specific level.

## Scope of AI use in this repository

**Project direction and validation** are human-authored. Decisions
about what the tool should do, how it is used with the camera and
calibration board, and whether a change is correct are made by the
maintainers. Results are checked against real images and hardware.

**Source code** (`apriltag_runner.py`, `camera_calibration.py`,
`check_container.py`) is ai-assisted: initial drafts, refactors, and
debugging suggestions may originate from an AI tool and are then
reviewed, run, and validated by a human contributor before merging.

**Container configuration** (`Dockerfile`, `docker-compose.yml`,
`entrypoint.sh`) is human-authored.

**Documentation** (`README.md`, `CONTRIBUTING.md`, and this file) is
ai-assisted: AI tools help with drafting and proofreading, and a human
contributor edits and validates the text before merging.

## Tools and models

AI assistance in this repository was provided by the following models:

| Provider | Models | Tool |
| --- | --- | --- |
| OpenAI | GPT-5, GPT-5.1, GPT-5.2, GPT-5.4 | ChatGPT |
| Anthropic | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Claude Code |

See the metadata block at the top of this file for the same list in
machine-readable form. Commits that include AI-assisted work may carry
a `Co-Authored-By` trailer naming the model. Where present, those
trailers are the record of which model contributed to a specific
change. Not every AI-assisted change is tagged this way, so the lists
above describe the models used across the project as a whole.

## Purpose of use

AI tools are used to:

- Draft, refactor, and debug the detection and calibration scripts.
- Draft, proofread, and improve documentation for clarity and
  accuracy.
- Generate commit messages and review suggestions.

## Input data / datasets

Only repository source files, public documentation, and
non-sensitive prompts are provided as input to AI models. No
sensitive, CUI, export-controlled, PII, HIPAA-regulated, or classified
data is submitted to external AI services, in accordance with BNL AI
Policy Security and Legality principles.

## Limitations and known biases

- LLM output may contain errors, hallucinations, or bias inherited
  from training data.
- AI-assisted code and content is reviewed and validated by a human
  contributor before it is merged. Because this project interacts with
  cameras and produces pose estimates, behavior should be verified on
  real images or hardware and not assumed from generated code.
- The models listed above do not have access to internal BNL systems,
  live data, or non-public information unless explicitly provided in a
  prompt.

## Reviewer disclaimer

AI-assisted content in this repository has been reviewed by the
project maintainers. Inclusion of AI-generated material does not
substitute for human accountability: the contributing staff member is
responsible for the correctness and appropriateness of every merged
change.
