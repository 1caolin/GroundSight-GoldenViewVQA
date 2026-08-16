#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoldenViewVQA Stage-2 View Reranking

"""

import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


# ============================================================
# 0. Configuration
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
).rstrip("/")

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
MAX_RETRY_OUTPUT_TOKENS = int(
    os.getenv("MAX_RETRY_OUTPUT_TOKENS", "4096")
)
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", "4"))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "3.0"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "360"))


SEND_IMAGE_DETAIL = (
    os.getenv("SEND_IMAGE_DETAIL", "false").lower() == "true"
)
IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "auto").lower()

VIEW_OVERRIDE_MODE = os.getenv("VIEW_OVERRIDE_MODE", "always").lower()


MAX_SAMPLES_ENV = os.getenv("MAX_SAMPLES", "")
MAX_SAMPLES = int(MAX_SAMPLES_ENV) if MAX_SAMPLES_ENV else None

RESUME = os.getenv("RESUME", "true").lower() == "true"


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        "/data2/ilearn/data2/caoruping/groundlm2026",
    )
)

GVQA_ROOT = PROJECT_ROOT / "datasets/GoldenViewVQA"
INPUT_FILE = GVQA_ROOT / "data/test_inputs.jsonl"

NUSCENES_ROOT = Path(
    os.getenv(
        "NUSCENES_ROOT",
        str(GVQA_ROOT / "data/nuscenes"),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs/test"),
    )
)


FIRST_STAGE_OUTPUT_FILE = Path(
    os.getenv(
        "FIRST_STAGE_OUTPUT_FILE",
        str(OUTPUT_DIR / "gpt56_view_rerank_from_existing_test.jsonl"),
    )
)
FIRST_STAGE_RAW_FILE = Path(
    os.getenv(
        "FIRST_STAGE_RAW_FILE",
        str(OUTPUT_DIR / "gpt56_view_rerank_from_existing_raw.jsonl"),
    )
)


STAGE2_OUTPUT_FILE = Path(
    os.getenv(
        "STAGE2_OUTPUT_FILE",
        str(OUTPUT_DIR / "gemini36_view_rerank_test.jsonl"),
    )
)
STAGE2_RAW_FILE = Path(
    os.getenv(
        "STAGE2_RAW_FILE",
        str(OUTPUT_DIR / "gemini36_view_rerank_raw.jsonl"),
    )
)


# ============================================================
# 2. Camera and label definitions
# ============================================================

CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

VALID_VIEWS = set(CAMERA_ORDER + ["NONE_OF_THE_ABOVE"])
VALID_ANSWERS = {"A", "B", "C", "D"}

ViewLiteral = Literal[
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "NONE_OF_THE_ABOVE",
]


class ViewRerankPrediction(BaseModel):
    predicted_view: ViewLiteral
    confidence: Literal["low", "medium", "high"]


# ============================================================
# 3. OpenAI-compatible client
# ============================================================

def build_client() -> OpenAI:
    api_key = "sk-"

    if not api_key:
        raise RuntimeError(
            "NO API key!"
        )

    print("Using base URL:", OPENAI_BASE_URL)
    print("Using endpoint:", f"{OPENAI_BASE_URL}/chat/completions")
    print("Using model:", MODEL_NAME)
    print("Image detail field enabled:", SEND_IMAGE_DETAIL)

    return OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        timeout=API_TIMEOUT,
        max_retries=0,
        default_headers={"Accept": "application/json"},
    )


client = build_client()


# ============================================================
# 4. Generic helpers
# ============================================================

def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return records


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def is_valid_prediction(obj: dict) -> bool:
    return (
        obj.get("predicted_view") in VALID_VIEWS
        and obj.get("predicted_answer_id") in VALID_ANSWERS
    )


def load_prediction_map(path: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}

    if not path.exists():
        return result

    for obj in read_jsonl(path):
        qid = obj.get("question_id")
        if qid and is_valid_prediction(obj):
            result[qid] = {
                "question_id": qid,
                "predicted_view": obj["predicted_view"],
                "predicted_answer_id": obj["predicted_answer_id"],
            }

    return result


def load_first_stage_predictions() -> Dict[str, dict]:

    predictions = load_prediction_map(FIRST_STAGE_OUTPUT_FILE)

    if FIRST_STAGE_RAW_FILE.exists():
        for raw in read_jsonl(FIRST_STAGE_RAW_FILE):
            qid = raw.get("question_id")
            parsed = raw.get("parsed_prediction") or {}

            if not qid:
                continue

            fallback = {
                "question_id": qid,
                "predicted_view": parsed.get("predicted_view"),
                "predicted_answer_id": parsed.get("predicted_answer_id"),
            }

            if qid not in predictions and is_valid_prediction(fallback):
                predictions[qid] = fallback

    return predictions


def load_completed_stage2(path: Path) -> Dict[str, dict]:
    return load_prediction_map(path)


# ============================================================
# 5. Stage-2 prompt
# ============================================================

def build_stage2_prompt(record: dict, frozen_answer_id: str) -> str:
    options = record["options"]
    frozen_answer_text = options[frozen_answer_id]

    return f"""
You are performing ONLY the second-stage visual evidence-source selection for
the GoldenViewVQA task.

The first-stage model has already selected the answer. The answer is frozen and
must not be changed.

Frozen answer ID:
{frozen_answer_id}

Frozen answer text:
{frozen_answer_text}

Your only task is to select the single supporting camera view.

Definition of supporting view:
Choose the single camera that contains the decisive visual evidence needed to
justify the frozen answer. Do not choose a camera merely because it contains
more objects, looks more salient, or is the default front camera.

Camera labels:
- CAM_FRONT
- CAM_FRONT_LEFT
- CAM_FRONT_RIGHT
- CAM_BACK
- CAM_BACK_LEFT
- CAM_BACK_RIGHT
- NONE_OF_THE_ABOVE

Camera convention:
- CAM_FRONT is directly ahead of the ego vehicle.
- CAM_FRONT_LEFT and CAM_FRONT_RIGHT are the forward-left and forward-right views.
- CAM_BACK is directly behind the ego vehicle.
- CAM_BACK_LEFT and CAM_BACK_RIGHT are the rear-left and rear-right views.
- Left and right are defined from the ego vehicle's coordinate system.
- Do not infer a camera name from the order of images in the prompt.

Inspect all six views independently. Internally evaluate:
1. Is the relevant object, road user, lane, signal, or spatial relation visible?
2. Does this view directly support the frozen answer?
3. Is the evidence decisive rather than merely contextual?
4. Does the view contradict the frozen answer?

Use NONE_OF_THE_ABOVE only when no single provided view contains sufficient
direct evidence.

Question type:
{record["question_group"]}

Question:
{record["question"]}

All answer options for context:
A: {options["A"]}
B: {options["B"]}
C: {options["C"]}
D: {options["D"]}

Return ONLY valid JSON. No markdown. No explanation.

The JSON must have exactly this shape:
{{
  "predicted_view": "CAM_FRONT",
  "confidence": "high"
}}

Do not output or change the answer ID.
""".strip()


# ============================================================
# 6. Image encoding and OpenAI Chat Completions messages
# ============================================================

def encode_image_data_url(image_path: Path) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    suffix = image_path.suffix.lower()
    if suffix not in mime_types:
        raise ValueError(f"Unsupported image type: {image_path}")

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_types[suffix]};base64,{image_base64}"


def build_stage2_messages(
    record: dict,
    frozen_answer_id: str,
) -> List[dict]:

    prompt = build_stage2_prompt(record, frozen_answer_id)

    content: List[dict] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for camera in CAMERA_ORDER:
        if camera not in record["views"]:
            raise KeyError(
                f"{record['question_id']} missing camera: {camera}"
            )

        image_path = (NUSCENES_ROOT / record["views"][camera]).resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for {record['question_id']} / {camera}: "
                f"{image_path}"
            )

        content.append(
            {
                "type": "text",
                "text": f"The following image is {camera}.",
            }
        )

        image_url = {
            "url": encode_image_data_url(image_path),
        }

        if SEND_IMAGE_DETAIL and IMAGE_DETAIL in {"auto", "low", "high"}:
            image_url["detail"] = IMAGE_DETAIL

        content.append(
            {
                "type": "image_url",
                "image_url": image_url,
            }
        )

    content.append(
        {
            "type": "text",
            "text": "Return only the required structured View prediction.",
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ============================================================
# 7. Chat Completions response helpers
# ============================================================

def extract_chat_text(response: Any) -> str:
    """读取 response.choices[0].message.content，兼容 str/list 两种返回。"""
    choices = get_field(response, "choices", []) or []
    if not choices:
        raise ValueError("API response has no choices")

    message = get_field(choices[0], "message", None)
    content = get_field(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if isinstance(part, dict):
                part_text = first_not_none(
                    part.get("text"),
                    part.get("content"),
                )
                if isinstance(part_text, str):
                    text_parts.append(part_text)
        return "".join(text_parts)

    return str(content) if content is not None else ""


def extract_usage(response: Any) -> dict:
    """读取 Chat Completions 的 prompt/completion/total tokens。"""
    usage = get_field(response, "usage", None)
    if usage is None:
        return {}

    prompt_tokens = first_not_none(
        get_field(usage, "prompt_tokens"),
        get_field(usage, "input_tokens"),
    )
    completion_tokens = first_not_none(
        get_field(usage, "completion_tokens"),
        get_field(usage, "output_tokens"),
    )
    total_tokens = get_field(usage, "total_tokens")

    prompt_details = first_not_none(
        get_field(usage, "prompt_tokens_details"),
        get_field(usage, "input_tokens_details"),
    )
    cached_tokens = first_not_none(
        get_field(prompt_details, "cached_tokens"),
        get_field(prompt_details, "cache_read_input_tokens"),
    )

    result = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if cached_tokens is not None:
        result["cached_input_tokens"] = cached_tokens
    return result


def extract_finish_reason(response: Any) -> Any:
    choices = get_field(response, "choices", []) or []
    if not choices:
        return None
    return get_field(choices[0], "finish_reason")


def estimate_cost(usage: dict) -> float:

    input_price = float(os.getenv("INPUT_PRICE_PER_MILLION", "0"))
    output_price = float(os.getenv("OUTPUT_PRICE_PER_MILLION", "0"))
    cached_price = float(
        os.getenv("CACHED_INPUT_PRICE_PER_MILLION", str(input_price))
    )

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cached_tokens = usage.get("cached_input_tokens") or 0
    uncached_input_tokens = max(input_tokens - cached_tokens, 0)

    return (
        uncached_input_tokens * input_price / 1_000_000
        + cached_tokens * cached_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


def clean_json_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]

    return text.strip()


def recover_prediction_from_broken_json(text: str) -> dict:

    view_names = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_BACK",
        "NONE_OF_THE_ABOVE",
    ]
    view_pattern = "|".join(re.escape(item) for item in view_names)

    view_match = re.search(
        rf"[\"']predicted_view[\"']\s*:\s*[\"']({view_pattern})[\"']",
        text or "",
        flags=re.IGNORECASE,
    )
    confidence_match = re.search(
        r"[\"']confidence[\"']\s*:\s*[\"'](low|medium|high)[\"']",
        text or "",
        flags=re.IGNORECASE,
    )

    if not view_match or not confidence_match:
        raise ValueError(
            "Could not recover predicted_view and confidence from model output"
        )

    return {
        "predicted_view": view_match.group(1).upper(),
        "confidence": confidence_match.group(1).lower(),
    }


def parse_view_prediction_text(text: str) -> tuple[dict, str, bool]:
    cleaned = clean_json_text(text)

    try:
        return json.loads(cleaned), cleaned, False
    except json.JSONDecodeError as json_error:
        try:
            recovered = recover_prediction_from_broken_json(text)
            return recovered, cleaned, True
        except Exception:
            raise json_error


def validate_view_prediction(data: dict) -> ViewRerankPrediction:
    model_validate = getattr(ViewRerankPrediction, "model_validate", None)
    if model_validate is not None:
        return model_validate(data)
    return ViewRerankPrediction.parse_obj(data)


def call_view_reranker(messages: List[dict]) -> dict:
    last_error: Any = None
    last_response: Any = None
    last_raw_output = ""
    last_clean_output = ""
    last_finish_reason: Any = None
    last_recovered = False

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            request_max_tokens = min(
                MAX_OUTPUT_TOKENS * (2 ** (attempt - 1)),
                MAX_RETRY_OUTPUT_TOKENS,
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=request_max_tokens,
                stream=False,
            )

            last_response = response
            last_raw_output = extract_chat_text(response)
            last_finish_reason = extract_finish_reason(response)

            if not last_raw_output.strip():
                raise ValueError("API returned empty assistant content")

            data, output_text, recovered = parse_view_prediction_text(
                last_raw_output
            )
            last_clean_output = output_text
            last_recovered = recovered
            parsed = validate_view_prediction(data)

            return {
                "success": True,
                "response": response,
                "parsed": parsed,
                "output_text": output_text,
                "raw_output": last_raw_output,
                "finish_reason": last_finish_reason,
                "recovered_json": recovered,
                "usage": extract_usage(response),
                "attempt": attempt,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            print(
                f"\nAPI attempt {attempt}/{MAX_API_RETRIES} failed: "
                f"{repr(exc)}"
            )
            print("finish_reason:", last_finish_reason)
            print("request max_tokens:", request_max_tokens)
            if last_raw_output:
                print(
                    "raw output preview:",
                    repr(last_raw_output[:1000]),
                )

            if last_finish_reason == "length":
                print(
                    "The response was truncated by max_tokens; "
                    "the next retry will use a larger limit."
                )

            if attempt < MAX_API_RETRIES:
                sleep_time = (
                    RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                    + random.uniform(0, 1.5)
                )
                print(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    print("View reranker failed; fallback to first-stage view.")

    return {
        "success": False,
        "response": last_response,
        "parsed": None,
        "output_text": last_clean_output,
        "raw_output": last_raw_output,
        "finish_reason": last_finish_reason,
        "recovered_json": last_recovered,
        "usage": extract_usage(last_response),
        "attempt": MAX_API_RETRIES,
        "error": repr(last_error),
    }


def choose_final_view(
    baseline_view: str,
    reranked_view: str,
    confidence: str,
) -> str:
    if VIEW_OVERRIDE_MODE == "always":
        return reranked_view

    if VIEW_OVERRIDE_MODE == "medium_high":
        return (
            reranked_view
            if confidence in {"medium", "high"}
            else baseline_view
        )

    if VIEW_OVERRIDE_MODE == "high_only":
        return reranked_view if confidence == "high" else baseline_view

    raise ValueError(
        "VIEW_OVERRIDE_MODE must be always, medium_high, or high_only"
    )


# ============================================================
# 8. Output canonicalization
# ============================================================

def canonicalize_output(records: List[dict], output_file: Path) -> None:
    predictions = load_prediction_map(output_file)

    missing = [
        record["question_id"]
        for record in records
        if record["question_id"] not in predictions
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} predictions are missing:\n"
            + "\n".join(missing)
        )

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            qid = record["question_id"]
            prediction = predictions[qid]
            clean = {
                "question_id": qid,
                "predicted_view": prediction["predicted_view"],
                "predicted_answer_id": prediction["predicted_answer_id"],
            }
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_FILE)
    first_stage = load_first_stage_predictions()

    if not first_stage:
        raise RuntimeError(
            "No valid first-stage predictions found. Expected at least one of:\n"
            f"  {FIRST_STAGE_OUTPUT_FILE}\n"
            f"  {FIRST_STAGE_RAW_FILE}"
        )

    missing_first_stage = [
        record["question_id"]
        for record in records
        if record["question_id"] not in first_stage
    ]

    if missing_first_stage:
        raise RuntimeError(
            "First-stage predictions are missing for "
            f"{len(missing_first_stage)} questions:\n"
            + "\n".join(missing_first_stage)
        )

    records_to_process = (
        records[:MAX_SAMPLES]
        if MAX_SAMPLES is not None
        else records
    )

    if RESUME:
        completed = load_completed_stage2(STAGE2_OUTPUT_FILE)
    else:
        completed = {}
        for path in (STAGE2_OUTPUT_FILE, STAGE2_RAW_FILE):
            if path.exists():
                path.unlink()

    remaining = [
        record
        for record in records_to_process
        if record["question_id"] not in completed
    ]

    print("\n" + "=" * 70)
    print("GoldenViewVQA Stage-2 View Reranking")
    print("OpenAI-compatible Chat Completions mode")
    print("The first-stage Answer is loaded from existing JSONL files.")
    print("=" * 70)
    print("Input file:", INPUT_FILE)
    print("First-stage official file:", FIRST_STAGE_OUTPUT_FILE)
    print("First-stage raw file:", FIRST_STAGE_RAW_FILE)
    print("Stage-2 output file:", STAGE2_OUTPUT_FILE)
    print("Stage-2 raw file:", STAGE2_RAW_FILE)
    print("Total input records:", len(records))
    print("Valid first-stage predictions:", len(first_stage))
    print("Samples to process:", len(records_to_process))
    print("Already completed stage-2:", len(completed))
    print("Remaining stage-2:", len(remaining))
    print("View override mode:", VIEW_OVERRIDE_MODE)
    print("=" * 70 + "\n")

    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_cost = 0.0
    changed_view_count = 0
    failed_rerank_count = 0

    progress = tqdm(remaining, desc="Stage-2 View reranking")

    for index, record in enumerate(progress, start=1):
        qid = record["question_id"]
        baseline = first_stage[qid]

        baseline_view = baseline["predicted_view"]
        frozen_answer_id = baseline["predicted_answer_id"]

        print("\n" + "=" * 70)
        print(f"Processing {index}/{len(remaining)}")
        print("question_id:", qid)
        print("question:", record["question"])
        print("frozen answer:", frozen_answer_id)
        print("first-stage view:", baseline_view)

        messages = build_stage2_messages(record, frozen_answer_id)
        rerank_result = call_view_reranker(messages)

        response = rerank_result["response"]
        parsed = rerank_result["parsed"]
        output_text = rerank_result["output_text"]
        raw_output = rerank_result.get("raw_output", "")
        finish_reason = rerank_result.get("finish_reason")
        usage = rerank_result["usage"]
        api_attempt = rerank_result["attempt"]

        if rerank_result["success"]:
            reranked_view = parsed.predicted_view
            confidence = parsed.confidence
            final_view = choose_final_view(
                baseline_view=baseline_view,
                reranked_view=reranked_view,
                confidence=confidence,
            )
        else:
            failed_rerank_count += 1
            reranked_view = baseline_view
            confidence = "low"
            final_view = baseline_view

        if final_view != baseline_view:
            changed_view_count += 1

        final_result = {
            "question_id": qid,
            "predicted_view": final_view,
            "predicted_answer_id": frozen_answer_id,
        }

        raw_record = {
            "question_id": qid,
            "model": MODEL_NAME,
            "api_mode": "chat.completions",
            "base_url": OPENAI_BASE_URL,
            "image_detail": IMAGE_DETAIL if SEND_IMAGE_DETAIL else None,
            "view_override_mode": VIEW_OVERRIDE_MODE,
            "first_stage": {
                "predicted_view": baseline_view,
                "predicted_answer_id": frozen_answer_id,
            },
            "stage2": {
                "success": rerank_result["success"],
                "response_id": get_field(response, "id"),
                "api_attempt": api_attempt,
                "finish_reason": finish_reason,
                "recovered_json": rerank_result.get("recovered_json", False),
                "raw_output": raw_output,
                "cleaned_output": output_text,
                "predicted_view": reranked_view,
                "confidence": confidence,
                "error": rerank_result["error"],
                "usage": usage,
            },
            "final_prediction": final_result,
            "estimated_request_cost_usd": estimate_cost(usage),
        }

        append_jsonl(STAGE2_OUTPUT_FILE, final_result)
        append_jsonl(STAGE2_RAW_FILE, raw_record)
        completed[qid] = final_result

        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        cached_tokens = usage.get("cached_input_tokens") or 0
        request_cost = estimate_cost(usage)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cached_tokens += cached_tokens
        total_cost += request_cost

        print("\nSTAGE 2 OUTPUT:")
        print(output_text)
        print("reranked view:", reranked_view)
        print("confidence:", confidence)
        print("finish_reason:", finish_reason)
        print("recovered JSON:", rerank_result.get("recovered_json", False))
        print("final view:", final_view)
        print("frozen answer:", frozen_answer_id)
        print("final JSON:", json.dumps(final_result, ensure_ascii=False))
        print("estimated cost:", f"${request_cost:.4f}")

        progress.set_postfix(
            baseline_view=baseline_view,
            reranked_view=reranked_view,
            final_view=final_view,
            answer=frozen_answer_id,
            cost=f"${total_cost:.2f}",
        )

    completed = load_completed_stage2(STAGE2_OUTPUT_FILE)

    if MAX_SAMPLES is None and len(completed) == len(records):
        canonicalize_output(records, STAGE2_OUTPUT_FILE)
        print("\nCanonicalized official submission file:")
        print(STAGE2_OUTPUT_FILE)
    elif MAX_SAMPLES is not None:
        print("\nPartial run detected; output is not a complete submission yet.")

    print("\n" + "=" * 70)
    print("Stage-2 View reranking finished")
    print("=" * 70)
    print("Completed:", len(completed), "/", len(records))
    print("Total input tokens:", total_input_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Total cached input tokens:", total_cached_tokens)
    print("Estimated total cost:", f"${total_cost:.4f}")
    print("Final submission file:", STAGE2_OUTPUT_FILE)
    print("Raw stage-2 file:", STAGE2_RAW_FILE)
    print(
        "Stage-2 changed view:",
        changed_view_count,
        "/",
        len(records_to_process),
    )
    print(
        "Stage-2 failed fallback:",
        failed_rerank_count,
        "/",
        len(records_to_process),
    )


if __name__ == "__main__":
    main()
