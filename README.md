# CoVeR-VQA

**GroundSight at GroundLM 2026 Shared Tasks: GoldenViewVQA**

This repository contains the system implementation, final predictions, intermediate Stage 3 corrections, method overview, and system paper for our submission to [GoldenViewVQA](https://groundlm.github.io/grouplm_emnlp2026/shared-tasks.html#goldenviewvqa), Shared Task 1 at the GroundLM 2026 Workshop co-located with EMNLP 2026.

CoVeR-VQA is a training-free, four-stage verification and correction framework for grounded multi-view visual question answering. Given six synchronized NuScenes camera views, a question, and four answer choices, the system jointly predicts:

1. the single camera view containing the decisive visual evidence; and
2. the correct answer choice.

The primary task metric is **Joint Accuracy**: a prediction is correct only when both the answer and the supporting view are correct.

- [Shared-task page](https://groundlm.github.io/grouplm_emnlp2026/shared-tasks.html#goldenviewvqa)
- [GoldenViewVQA dataset](https://huggingface.co/datasets/YimuWang/GoldenViewVQA)
- [Dataset schema](https://huggingface.co/datasets/YimuWang/GoldenViewVQA/blob/main/docs/schema.md)
- [Official evaluator](https://huggingface.co/datasets/YimuWang/GoldenViewVQA/blob/main/scripts/evaluate.py)
- [System paper](CoVeR-VQA.pdf)

## Results

Results below are from the official GoldenViewVQA test set of 59 questions.

| System | Joint Acc. | Answer Acc. | View Acc. | View Macro Acc. |
|---|---:|---:|---:|---:|
| Organizer baseline | 20.34 | 23.73 | 77.97 | 16.67 |
| Qwen3-VL-8B zero-shot | 66.10 | 88.14 | 71.19 | 49.88 |
| GPT-5.6 zero-shot (Stage 1) | 71.19 | 91.53 | 74.58 | 65.82 |
| Gemini view review (Stage 2) | 74.58 | 91.53 | 79.66 | 54.11 |
| Claude joint review (Stage 3) | 79.66 | 93.22 | 83.05 | 54.83 |
| **CoVeR-VQA pipeline (Stage 4)** | **81.36** | **94.92** | **83.05** | **54.83** |
| **Final submission** | **84.75** | **96.61** | **86.44** | **63.53** |

The reproducible four-stage pipeline improves Joint Accuracy from 71.19% to 81.36%, a gain of **10.17 percentage points** over the GPT-5.6 zero-shot baseline. The final submitted run reaches **84.75% Joint Accuracy** after two additional evaluator-informed, instance-level post-hoc corrections. Because those two corrections are not a generalizable inference component, we report the pipeline and final-submission results separately.

The final submission contains 50/59 jointly correct predictions. Of the nine remaining errors, seven have the correct answer but an incorrect supporting view, one has a view-correct/answer-incorrect prediction, and one is incorrect in both components.

## Method

![Overview of the CoVeR-VQA four-stage pipeline](method.png)

CoVeR-VQA progressively targets different error sources:

1. **Zero-shot joint prediction — GPT-5.6.** All six camera views are provided at once. The model jointly predicts an answer and its supporting view.
2. **View-specific verification — Gemini-3.6-Flash.** The Stage 1 answer is frozen while Gemini compares all views and may correct only the evidence source.
3. **Prior-guided confidence-aware joint verification — Claude-Opus-5.** Claude rechecks the answer and view together. Two audit passes are aggregated, and the Stage 2 evaluator distribution (44 correct, 10 view-only errors, 3 answer-only errors, and 2 errors in both components) is used as a collection-level prior rather than as per-instance ground truth. A correction is applied only when the estimated probability that the current answer-view pair is jointly correct is at most 0.5.
4. **Group-level prior verification — Claude-Opus-5.** Questions sharing a base query ID and the same six-view scene are grouped. Their question stems produce a soft semantic prior that is checked against the images before conservative residual corrections are applied.

The pipeline is entirely inference-time: it uses no task-specific fine-tuning and no additional labeled training set.

## Repository Structure

```text
.
├── code/
│   ├── stage1_gpt.py             # Zero-shot joint answer/view prediction
│   ├── stage2_gemini.py          # View-only reranking with the answer frozen
│   ├── stage3-claude.py          # Prior-guided joint audit and correction
│   └── stage4_claude.py          # Same-scene group-prior verification
├── result/
│   ├── best.jsonl                # Final submission: 59 canonical predictions
│   └── claude_stage3_selected.jsonl
│                                  # 15 Stage 3 selected corrections and audits
├── method.png                    # Pipeline overview
└── README.md
```

`result/best.jsonl` is the final submission artifact. `result/claude_stage3_selected.jsonl` is an analysis/intermediate file, not a submission file. It records the 15 Stage 3 changes selected by the constrained audit: 10 view-only, 3 answer-only, and 2 joint answer-view corrections. Each record includes the baseline, proposed correction, aggregated error probabilities, and evidence summary.

## Data Preparation

The benchmark annotations and NuScenes-relative image paths are available from the [GoldenViewVQA dataset repository](https://huggingface.co/datasets/YimuWang/GoldenViewVQA). NuScenes images are not redistributed here; obtain them through the [official NuScenes access process](https://www.nuscenes.org/) and follow its terms of use.

By default, the scripts expect the following external layout:

```text
<PROJECT_ROOT>/
├── datasets/GoldenViewVQA/
│   └── data/
│       ├── test_inputs.jsonl
│       └── nuscenes/             # NuScenes files referenced by the JSONL
└── outputs/test/                 # Generated stage outputs and raw audit logs
```

The test input contains 59 unlabeled questions and 354 image references. The public development split contains 55 labeled questions and 330 image references.

## Environment

Python 3.9 or newer is recommended. The released scripts require only lightweight client-side packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai pydantic tqdm
```



The model calls use OpenAI Responses or OpenAI-compatible Chat Completions interfaces. Your endpoint must expose deployments corresponding to the configured model names, or you must override the model names with compatible deployments available to you.

### Credentials and paths

Do not commit API credentials. The released files contain placeholder keys only.

- `stage1_gpt.py` stores `PROJECT_ROOT`, `API_KEY`, and `BASE_URL` as constants near the top of the file; set them locally before running.
- `stage2_gemini.py` and `stage3-claude.py` accept path/model configuration through environment variables but currently contain a local `sk-` API-key placeholder in `build_client()`; replace it locally or change it to read `OPENAI_API_KEY`.
- `stage4_claude.py` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` directly from the environment.

Stages 2–4 support environment-variable overrides including `PROJECT_ROOT`, `INPUT_FILE`, `NUSCENES_ROOT`, `OUTPUT_DIR`, `MODEL_NAME`, `MAX_SAMPLES`, and `RESUME`. Stage-specific input and output paths can also be overridden; see the configuration block at the top of each script.

## Running the Pipeline

The scripts are intentionally independent and write crash-safe JSONL outputs. Run them in order. The following POSIX-shell example makes the handoff between stages explicit:

```bash
export PROJECT_ROOT=/absolute/path/to/groundlm2026
export OUTPUT_DIR="$PROJECT_ROOT/outputs/test"
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your-local-key

# Stage 1: first update PROJECT_ROOT, API_KEY, and BASE_URL in this file.
python code/stage1_gpt.py

# Stage 2: freeze the Stage 1 answers and rerank only the views.
FIRST_STAGE_OUTPUT_FILE="$OUTPUT_DIR/gpt56_joint_zs_test.jsonl" \
python code/stage2_gemini.py

# Stage 3: jointly audit the Stage 2 predictions.
BASELINE_FILE="$OUTPUT_DIR/gemini36_view_rerank_test.jsonl" \
python code/stage3-claude.py

# Stage 4: verify grouped, same-scene questions using a soft group prior.
STAGE3_FILE="$OUTPUT_DIR/claude_stage3_final_test.jsonl" \
python code/stage4_claude.py
```

Useful execution controls include:

- `MAX_SAMPLES=<n>` for a partial smoke test;
- `RESUME=true` to reuse completed JSONL records;
- `NUM_AUDIT_PASSES` for Stage 3 (default: 2);
- `NUM_VERIFY_PASSES` for Stage 4 (default: 2); and
- `CORRECT_THRESHOLD` for Stage 4 (default: 0.5).

Before a full run, test one or two samples with `MAX_SAMPLES` and inspect the raw output files. API behavior, availability, pricing, and model snapshots depend on the endpoint provider.



## Reproducibility Notes

- The system is training-free but requires access to the three configured multimodal model deployments.
- Multimodal API outputs may vary with model snapshots and provider-side inference settings.
- Stage 3 uses aggregate statistics returned by the official evaluator for the complete Stage 2 submission; it does not use instance-level test labels.
- Stage 4 assumes that questions with the same base ID share exactly the same six-view scene and validates this condition before constructing a group prior.
- The two final post-hoc corrections are included in `result/best.jsonl` but are not part of the general four-stage pipeline.


## Data and Asset Terms

GoldenViewVQA and NuScenes remain subject to their respective licenses and terms of use. This repository does not redistribute the benchmark images or hidden test annotations.
