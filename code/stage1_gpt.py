import os
import json
import base64
import time
import random
from pathlib import Path
from typing import Literal

from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI



# ============================================================
# 0. OpenAI configuration
# ============================================================

MODEL_NAME = "gpt-5.6"

# GPT-5.6 reasoning effort:
# none / low / medium / high / xhigh / max
#
# 第一版建议 high。
REASONING_EFFORT = "high"

# Image detail:
IMAGE_DETAIL = "auto"
MAX_OUTPUT_TOKENS = 1024
# API retry
MAX_API_RETRIES = 4

RETRY_BASE_SLEEP = 3.0

# OpenAI client timeout
API_TIMEOUT = 180.0


# ============================================================
# 1. Path configuration
# ============================================================

PROJECT_ROOT = Path(
    "/data2/ilearn/data2/caoruping/groundlm2026"
)

GVQA_ROOT = (
    PROJECT_ROOT /
    "datasets/GoldenViewVQA"
)

INPUT_FILE = (
    GVQA_ROOT /
    "data/test_inputs.jsonl"
)

NUSCENES_ROOT = (
    GVQA_ROOT /
    "data/nuscenes"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs/test"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "gpt56_joint_zs_test.jsonl"
)

RAW_OUTPUT_FILE = (
    OUTPUT_DIR /
    "gpt56_joint_zs_test_raw.jsonl"
)


# ============================================================
# 2. Runtime configuration
# ============================================================

CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

VALID_VIEWS = set(
    CAMERA_ORDER +
    ["NONE_OF_THE_ABOVE"]
)

VALID_ANSWERS = {
    "A",
    "B",
    "C",
    "D",
}

RESUME = True
MAX_SAMPLES = None


# ============================================================
# 3. OpenAI client
# ============================================================

API_KEY = "sk-"

BASE_URL = ""


def build_openai_client():

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=API_TIMEOUT,
        max_retries=0,
    )

    print(
        "Using custom OpenAI base URL:",
        BASE_URL,
    )

    return client


client = build_openai_client()



# ============================================================
# 4. Structured output schema
# ============================================================

class GoldenViewPrediction(BaseModel):

    predicted_view: Literal[
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "NONE_OF_THE_ABOVE",
    ]

    predicted_answer_id: Literal[
        "A",
        "B",
        "C",
        "D",
    ]


# ============================================================
# 5. JSONL utilities
# ============================================================

def read_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def load_completed(path):

    completed = {}

    path = Path(path)

    if not path.exists():
        return completed

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

                qid = obj["question_id"]

                if (
                    obj.get(
                        "predicted_view"
                    )
                    in VALID_VIEWS
                    and
                    obj.get(
                        "predicted_answer_id"
                    )
                    in VALID_ANSWERS
                ):

                    completed[qid] = obj

            except Exception:

                # Ignore incomplete trailing line.
                continue

    return completed


# ============================================================
# 6. Prompt
# ============================================================

def build_prompt(record):

    options = record["options"]

    prompt = f"""
You are solving a multi-view visual question answering task
for an autonomous-driving scene.

You are given six synchronized camera images captured at the
same timestamp.

Your task requires TWO coupled decisions:

1. Select the SINGLE camera view that provides the clearest
   and most direct visual evidence needed to answer the question.

2. Select the correct multiple-choice answer.

Allowed camera labels:

CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT
NONE_OF_THE_ABOVE

Important rules:

- Inspect all six synchronized camera views before deciding.

- Select the camera that provides the most direct visual
  evidence required by the question.

- Do not select a camera merely because it contains visually
  salient objects.

- If multiple views contain related evidence, select the
  SINGLE view that provides the clearest and most direct
  support for the reasoning required by the question.

- Adjacent cameras may contain overlapping objects.
  Distinguish carefully between neighboring views.

- Use NONE_OF_THE_ABOVE only when none of the six provided
  views supplies sufficient direct visual evidence.

- Base the answer primarily on visible evidence in the
  provided images rather than generic driving priors.

- The predicted camera view and predicted answer must be
  mutually consistent.

Question type:
{record["question_group"]}

Question:
{record["question"]}

Options:

A: {options["A"]}

B: {options["B"]}

C: {options["C"]}

D: {options["D"]}

First determine which answer option is best supported by the
six synchronized images.

Then determine which single camera view provides the clearest
and most direct visual support for that answer.

Do not output reasoning or explanations.

Return only the required structured prediction.
""".strip()

    return prompt


# ============================================================
# 7. Image encoding
# ============================================================

def encode_image_data_url(
    image_path,
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Missing image:\n{image_path}"
        )

    suffix = (
        image_path
        .suffix
        .lower()
    )

    if suffix in [
        ".jpg",
        ".jpeg",
    ]:

        mime_type = "image/jpeg"

    elif suffix == ".png":

        mime_type = "image/png"

    elif suffix == ".webp":

        mime_type = "image/webp"

    elif suffix == ".gif":

        mime_type = "image/gif"

    else:

        raise ValueError(
            f"Unsupported image type: "
            f"{image_path}"
        )

    with open(
        image_path,
        "rb",
    ) as f:

        image_bytes = f.read()

    image_base64 = (
        base64
        .b64encode(image_bytes)
        .decode("utf-8")
    )

    return (
        f"data:{mime_type};base64,"
        f"{image_base64}"
    )


# ============================================================
# 8. Build Responses API input
# ============================================================

def build_input(
    record,
):

    prompt = build_prompt(
        record
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    # --------------------------------------------------------
    # Six synchronized camera views
    # --------------------------------------------------------

    for camera in CAMERA_ORDER:

        if camera not in record["views"]:

            raise KeyError(
                f"{record['question_id']} "
                f"missing camera: {camera}"
            )

        image_path = (
            Path(NUSCENES_ROOT)
            /
            record["views"][camera]
        ).resolve()

        if not image_path.exists():

            raise FileNotFoundError(
                f"Missing image:\n"
                f"{image_path}"
            )

        # Explicit camera-image association.
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"The following image is "
                    f"{camera}."
                ),
            }
        )

        image_data_url = (
            encode_image_data_url(
                image_path
            )
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url,
                "detail": IMAGE_DETAIL,
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "Inspect all six views and "
                "return the final structured prediction."
            ),
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ============================================================
# 9. API usage extraction
# ============================================================

def extract_usage(
    response,
):

    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        return {}

    result = {
        "input_tokens":
            getattr(
                usage,
                "input_tokens",
                None,
            ),

        "output_tokens":
            getattr(
                usage,
                "output_tokens",
                None,
            ),

        "total_tokens":
            getattr(
                usage,
                "total_tokens",
                None,
            ),
    }

    input_details = getattr(
        usage,
        "input_tokens_details",
        None,
    )

    if input_details is not None:

        result[
            "cached_input_tokens"
        ] = getattr(
            input_details,
            "cached_tokens",
            None,
        )

    return result


# ============================================================
# 10. Approximate cost
# ============================================================

def estimate_cost(
    usage,
):

    input_tokens = (
        usage.get(
            "input_tokens"
        )
        or 0
    )

    output_tokens = (
        usage.get(
            "output_tokens"
        )
        or 0
    )

    cached_tokens = (
        usage.get(
            "cached_input_tokens"
        )
        or 0
    )

    # GPT-5.6 Sol:
    #
    # Input:
    #   $5 / 1M
    #
    # Cached input:
    #   $0.50 / 1M
    #
    # Output:
    #   $30 / 1M
    #
    # This is an approximate estimate.
    #
    # We treat cached tokens at the cached-input rate.

    uncached_input_tokens = max(
        input_tokens
        - cached_tokens,
        0,
    )

    cost = (
        uncached_input_tokens
        * 5.0
        / 1_000_000
    )

    cost += (
        cached_tokens
        * 0.50
        / 1_000_000
    )

    cost += (
        output_tokens
        * 30.0
        / 1_000_000
    )

    return cost


# ============================================================
# 11. GPT-5.6 inference
# ============================================================

def call_gpt56(
    messages,
):

    last_error = None

    for attempt in range(
        1,
        MAX_API_RETRIES + 1,
    ):

        try:

            response = (
                client.responses.parse(
                    model=MODEL_NAME,

                    input=messages,

                    reasoning={
                        "effort":
                            REASONING_EFFORT
                    },

                    max_output_tokens=(
                        MAX_OUTPUT_TOKENS
                    ),

                    text_format=(
                        GoldenViewPrediction
                    ),
                )
            )

            parsed = (
                response.output_parsed
            )

            output_text = (
                response.output_text
            )

            if parsed is None:

                raise RuntimeError(
                    "GPT-5.6 returned no "
                    "structured prediction.\n"
                    f"Raw output:\n"
                    f"{output_text}"
                )

            usage = extract_usage(
                response
            )

            return (
                response,
                parsed,
                output_text,
                usage,
                attempt,
            )

        except Exception as e:

            last_error = e

            print(
                f"\nAPI attempt "
                f"{attempt}/"
                f"{MAX_API_RETRIES} failed:"
            )

            print(
                repr(e)
            )

            if attempt >= MAX_API_RETRIES:

                break

            sleep_time = (
                RETRY_BASE_SLEEP
                * (2 ** (attempt - 1))
                + random.uniform(
                    0,
                    1.5,
                )
            )

            print(
                f"Retrying in "
                f"{sleep_time:.1f}s..."
            )

            time.sleep(
                sleep_time
            )

    raise RuntimeError(
        "GPT-5.6 API failed after "
        f"{MAX_API_RETRIES} attempts."
    ) from last_error


# ============================================================
# 12. Canonicalize final output
# ============================================================

def canonicalize_predictions(
    records,
    output_file,
):

    predictions = load_completed(
        output_file
    )

    missing = [
        record["question_id"]
        for record in records
        if record["question_id"]
        not in predictions
    ]

    if missing:

        raise RuntimeError(
            f"{len(missing)} predictions "
            f"are still missing:\n"
            + "\n".join(missing)
        )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fw:

        for record in records:

            qid = record[
                "question_id"
            ]

            result = predictions[
                qid
            ]

            clean_result = {
                "question_id":
                    qid,

                "predicted_view":
                    result[
                        "predicted_view"
                    ],

                "predicted_answer_id":
                    result[
                        "predicted_answer_id"
                    ],
            }

            fw.write(
                json.dumps(
                    clean_result,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 13. Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = read_jsonl(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Limit samples if requested
    # --------------------------------------------------------

    if MAX_SAMPLES is not None:

        records_to_process = (
            records[
                :MAX_SAMPLES
            ]
        )

    else:

        records_to_process = records

    print(
        "\n========================================"
    )

    print(
        "GoldenViewVQA GPT-5.6 Zero-shot"
    )

    print(
        "========================================"
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Reasoning effort:",
        REASONING_EFFORT,
    )

    print(
        "Image detail:",
        IMAGE_DETAIL,
    )

    print(
        "Input:",
        INPUT_FILE,
    )

    print(
        "Total test samples:",
        len(records),
    )

    print(
        "Samples to process:",
        len(records_to_process),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Raw output:",
        RAW_OUTPUT_FILE,
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if RESUME:

        completed = load_completed(
            OUTPUT_FILE
        )

    else:

        completed = {}

        if OUTPUT_FILE.exists():

            OUTPUT_FILE.unlink()

        if RAW_OUTPUT_FILE.exists():

            RAW_OUTPUT_FILE.unlink()

    print(
        "Already completed:",
        len(completed),
    )

    remaining = [
        record
        for record in records_to_process
        if record["question_id"]
        not in completed
    ]

    print(
        "Remaining:",
        len(remaining),
    )

    print()

    # --------------------------------------------------------
    # Runtime statistics
    # --------------------------------------------------------

    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_cost = 0.0

    # --------------------------------------------------------
    # Crash-safe append
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as prediction_fw, open(
        RAW_OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as raw_fw:

        progress = tqdm(
            remaining,
            desc="GPT-5.6 inference",
        )

        for index, record in enumerate(
            progress,
            start=1,
        ):

            qid = record[
                "question_id"
            ]

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Processing "
                f"{index}/"
                f"{len(remaining)}"
            )

            print(
                "question_id:",
                qid,
            )

            print(
                "question:",
                record["question"],
            )

            # ------------------------------------------------
            # Build multimodal input
            # ------------------------------------------------

            messages = build_input(
                record
            )

            # ------------------------------------------------
            # API inference
            # ------------------------------------------------

            (
                response,
                parsed,
                output_text,
                usage,
                api_attempt,
            ) = call_gpt56(
                messages
            )

            # ------------------------------------------------
            # Extract prediction
            # ------------------------------------------------

            predicted_view = (
                parsed.predicted_view
            )

            predicted_answer = (
                parsed.predicted_answer_id
            )

            # ------------------------------------------------
            # Usage
            # ------------------------------------------------

            input_tokens = (
                usage.get(
                    "input_tokens",
                    0,
                )
                or 0
            )

            output_tokens = (
                usage.get(
                    "output_tokens",
                    0,
                )
                or 0
            )

            cached_tokens = (
                usage.get(
                    "cached_input_tokens",
                    0,
                )
                or 0
            )

            request_cost = (
                estimate_cost(
                    usage
                )
            )

            total_input_tokens += (
                input_tokens
            )

            total_output_tokens += (
                output_tokens
            )

            total_cached_tokens += (
                cached_tokens
            )

            total_cost += (
                request_cost
            )

            # ------------------------------------------------
            # Official prediction
            # ------------------------------------------------

            result = {
                "question_id":
                    qid,

                "predicted_view":
                    predicted_view,

                "predicted_answer_id":
                    predicted_answer,
            }

            # ------------------------------------------------
            # Raw record
            # ------------------------------------------------

            raw_record = {
                "question_id":
                    qid,

                "model":
                    MODEL_NAME,

                "reasoning_effort":
                    REASONING_EFFORT,

                "image_detail":
                    IMAGE_DETAIL,

                "api_attempt":
                    api_attempt,

                "response_id":
                    getattr(
                        response,
                        "id",
                        None,
                    ),

                "raw_output":
                    output_text,

                "parsed_prediction":
                    {
                        "predicted_view":
                            predicted_view,

                        "predicted_answer_id":
                            predicted_answer,
                    },

                "usage":
                    usage,

                "estimated_request_cost_usd":
                    request_cost,
            }

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            prediction_fw.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            prediction_fw.flush()

            raw_fw.write(
                json.dumps(
                    raw_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            raw_fw.flush()

            completed[
                qid
            ] = result

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            print(
                "\nGPT-5.6 OUTPUT:"
            )

            print(
                output_text
            )

            print(
                "\nPARSED:"
            )

            print(
                "predicted_view:",
                predicted_view,
            )

            print(
                "predicted_answer_id:",
                predicted_answer,
            )

            print(
                "\nUSAGE:"
            )

            print(
                "input_tokens:",
                input_tokens,
            )

            print(
                "output_tokens:",
                output_tokens,
            )

            print(
                "cached_input_tokens:",
                cached_tokens,
            )

            print(
                "estimated cost:",
                f"${request_cost:.4f}",
            )

            print(
                "=" * 60
            )

            progress.set_postfix(
                view=predicted_view,
                answer=predicted_answer,
                cost=f"${total_cost:.2f}",
            )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "Inference finished"
    )

    print(
        "========================================"
    )

    completed = load_completed(
        OUTPUT_FILE
    )

    print(
        "Completed:",
        len(completed),
        "/",
        len(records),
    )

    # --------------------------------------------------------
    # If complete test set:
    # canonicalize according to test_inputs.jsonl order
    # --------------------------------------------------------

    if (
        MAX_SAMPLES is None
        and
        len(completed)
        == len(records)
    ):

        canonicalize_predictions(
            records,
            OUTPUT_FILE,
        )

        print(
            "\nCanonicalized prediction file:"
        )

        print(
            OUTPUT_FILE
        )

    elif MAX_SAMPLES is not None:

        print(
            "\nPartial run detected."
        )

        print(
            "Prediction file is NOT yet a "
            "complete official test submission."
        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "API statistics"
    )

    print(
        "========================================"
    )

    print(
        "Total input tokens:",
        total_input_tokens,
    )

    print(
        "Total output tokens:",
        total_output_tokens,
    )

    print(
        "Total cached input tokens:",
        total_cached_tokens,
    )

    print(
        "Estimated total cost:",
        f"${total_cost:.4f}",
    )

    print(
        "\nFinal prediction file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nRaw model outputs:"
    )

    print(
        RAW_OUTPUT_FILE
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()