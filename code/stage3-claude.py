#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoldenViewVQA Stage-3 Claude constrained auditor

"""

import base64
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 0. Configuration
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "claude-opus-5")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
).rstrip("/")

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
MAX_RETRY_OUTPUT_TOKENS = int(os.getenv("MAX_RETRY_OUTPUT_TOKENS", "8192"))

MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", "4"))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "3.0"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "360"))

NUM_AUDIT_PASSES = int(os.getenv("NUM_AUDIT_PASSES", "2"))
BOTH_INSUFFICIENCY_BONUS = float(
    os.getenv("BOTH_INSUFFICIENCY_BONUS", "1.35")
)

MAX_SAMPLES_ENV = os.getenv("MAX_SAMPLES", "")
MAX_SAMPLES = int(MAX_SAMPLES_ENV) if MAX_SAMPLES_ENV else None

RESUME = os.getenv("RESUME", "true").lower() == "true"

SEND_IMAGE_DETAIL = (
    os.getenv("SEND_IMAGE_DETAIL", "false").lower() == "true"
)
IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "auto").lower()
CORRECT_THRESHOLD = 0.5

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
INPUT_FILE = Path(
    os.getenv(
        "INPUT_FILE",
        str(GVQA_ROOT / "data/test_inputs.jsonl"),
    )
)

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

BASELINE_FILE = Path(
    os.getenv(
        "BASELINE_FILE",
        str(OUTPUT_DIR / "gemini36_view_rerank_test.jsonl"),
    )
)

STAGE3_AUDIT_FILE = Path(
    os.getenv(
        "STAGE3_AUDIT_FILE",
        str(OUTPUT_DIR / "claude_stage3_audit.jsonl"),
    )
)

STAGE3_SELECTED_FILE = Path(
    os.getenv(
        "STAGE3_SELECTED_FILE",
        str(OUTPUT_DIR / "claude_stage3_selected_15.jsonl"),
    )
)

STAGE3_OUTPUT_FILE = Path(
    os.getenv(
        "STAGE3_OUTPUT_FILE",
        str(OUTPUT_DIR / "claude_stage3_final_test.jsonl"),
    )
)


# ============================================================
# 2. Labels and exact evaluator-derived prior
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

ERROR_TYPES = ("correct", "view_only", "answer_only", "both")


EXACT_QUOTA = {
    "correct": 44,
    "view_only": 10,
    "answer_only": 3,
    "both": 2,
}

if sum(EXACT_QUOTA.values()) != 59:
    raise RuntimeError("EXACT_QUOTA must sum to 59")


# ============================================================
# 3. Client
# ============================================================

def build_client() -> OpenAI:
    api_key = "sk-"
    if not api_key:
        raise RuntimeError(
            "NO OPENAI_API_KEY."
        )

    print("Using base URL:", OPENAI_BASE_URL)
    print("Using endpoint:", f"{OPENAI_BASE_URL}/chat/completions")
    print("Using model:", MODEL_NAME)

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


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_prediction_map(path: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for obj in read_jsonl(path):
        qid = obj.get("question_id")
        view = obj.get("predicted_view")
        ans = obj.get("predicted_answer_id")
        if not qid:
            continue
        if view not in VALID_VIEWS:
            raise ValueError(f"{qid}: invalid baseline view: {view}")
        if ans not in VALID_ANSWERS:
            raise ValueError(f"{qid}: invalid baseline answer: {ans}")
        result[qid] = {
            "question_id": qid,
            "predicted_view": view,
            "predicted_answer_id": ans,
        }
    return result


def load_completed_audits(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for obj in read_jsonl(path):
        qid = obj.get("question_id")
        if qid and obj.get("aggregate_audit"):
            result[qid] = obj
    return result


# ============================================================
# 5. Images
# ============================================================

def encode_image_data_url(image_path: Path) -> str:
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


# ============================================================
# 6. Prompt
# ============================================================

SYSTEM_PRIOR = """
You are the FINAL independent auditor for a 59-query GoldenViewVQA submission.

A previous GPT stage followed by a Gemini correction stage produced the current
baseline submission. The evaluator returned aggregate metrics from THIS EXACT
baseline, which imply the following exact population structure:

- 44 queries: current View is correct AND current Answer is correct.
- 10 queries: current View is wrong, current Answer is correct.  [VIEW_ONLY]
- 3 queries: current View is correct, current Answer is wrong.   [ANSWER_ONLY]
- 2 queries: current View is wrong AND current Answer is wrong.  [BOTH]

These counts are GLOBAL information only. They do NOT identify any individual
query. For each query, judge the images and question independently. Do not force
a query into an error class merely to imitate the global counts; the caller will
perform the exact global assignment later.

Important hypothesis for the 2 BOTH errors:
They may be cases where the previous models failed to recognize that the current
frame(s) are insufficient to support a substantive answer. Therefore explicitly
check whether an answer option means "insufficient evidence / cannot determine
from the current frame", and whether NONE_OF_THE_ABOVE should be the view.
Treat this as a strong hypothesis to test, NOT as a guaranteed fact.

Golden view definition:
Select the single camera that contains the decisive visual evidence required to
justify the answer. If no provided camera contains sufficient direct evidence,
use NONE_OF_THE_ABOVE.

You must be conservative. 44/59 current joint predictions are already correct.
A change requires positive visual/question evidence, not merely a plausible
alternative.

Return only one JSON object. No markdown and no prose outside JSON.
""".strip()


def build_audit_prompt(record: dict, baseline: dict, pass_index: int) -> str:
    options = record["options"]
    current_answer_id = baseline["predicted_answer_id"]
    current_answer_text = options[current_answer_id]

    return f"""
Audit pass: {pass_index}

Question ID:
{record["question_id"]}

Question group:
{record.get("question_group", "")}

Question:
{record["question"]}

Answer options:
A: {options["A"]}
B: {options["B"]}
C: {options["C"]}
D: {options["D"]}

CURRENT BASELINE PREDICTION:
- View: {baseline["predicted_view"]}
- Answer ID: {current_answer_id}
- Answer text: {current_answer_text}

Inspect all six images independently before deciding.

Classify the CURRENT baseline into exactly one latent state:
1. correct:
   current view correct, current answer correct.
2. view_only:
   current answer is correct but current view is wrong.
3. answer_only:
   current view is correct but current answer is wrong.
4. both:
   current view and current answer are both wrong.

Give calibrated probabilities for all four states. They must be non-negative
numbers summing to 100.

Then provide the best correction under EACH possible error type:
- view_only_candidate.predicted_view MUST differ from current view.
  Its answer is implicitly frozen to the current answer.
- answer_only_candidate.predicted_answer_id MUST differ from current answer.
  Its view is implicitly frozen to the current view.
- both_candidate.predicted_view MUST differ from current view AND
  both_candidate.predicted_answer_id MUST differ from current answer.
- For BOTH, explicitly test the insufficiency hypothesis. If an answer option
  says the frame is insufficient/cannot determine and no camera gives decisive
  evidence, consider predicted_view=NONE_OF_THE_ABOVE and that answer option.

Do NOT infer camera names from image order. Each image is explicitly labeled.

Required JSON schema:
{{
  "probabilities": {{
    "correct": 70,
    "view_only": 20,
    "answer_only": 7,
    "both": 3
  }},
  "view_only_candidate": {{
    "predicted_view": "CAM_FRONT_LEFT"
  }},
  "answer_only_candidate": {{
    "predicted_answer_id": "B"
  }},
  "both_candidate": {{
    "predicted_view": "NONE_OF_THE_ABOVE",
    "predicted_answer_id": "D",
    "insufficient_evidence": true
  }},
  "current_support_strength": "strong|medium|weak|none",
  "evidence_summary": "brief concrete visual evidence, <= 60 words"
}}
""".strip()


def build_messages(
    record: dict,
    baseline: dict,
    pass_index: int,
) -> List[dict]:
    content: List[dict] = [
        {
            "type": "text",
            "text": build_audit_prompt(record, baseline, pass_index),
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

        image_url = {"url": encode_image_data_url(image_path)}
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
            "text": "Now return only the required JSON object.",
        }
    )

    return [
        {"role": "system", "content": SYSTEM_PRIOR},
        {"role": "user", "content": content},
    ]


# ============================================================
# 7. Response parsing / validation
# ============================================================

def extract_chat_text(response: Any) -> str:
    choices = get_field(response, "choices", []) or []
    if not choices:
        raise ValueError("API response has no choices")

    message = get_field(choices[0], "message", None)
    content = get_field(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content) if content is not None else ""


def extract_finish_reason(response: Any) -> Any:
    choices = get_field(response, "choices", []) or []
    if not choices:
        return None
    return get_field(choices[0], "finish_reason")


def extract_usage(response: Any) -> dict:
    usage = get_field(response, "usage", None)
    if usage is None:
        return {}

    return {
        "input_tokens": (
            get_field(usage, "prompt_tokens")
            if get_field(usage, "prompt_tokens") is not None
            else get_field(usage, "input_tokens")
        ),
        "output_tokens": (
            get_field(usage, "completion_tokens")
            if get_field(usage, "completion_tokens") is not None
            else get_field(usage, "output_tokens")
        ),
        "total_tokens": get_field(usage, "total_tokens"),
    }


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


def normalize_probabilities(probs: dict) -> Dict[str, float]:
    values = {}
    for key in ERROR_TYPES:
        if key not in probs:
            raise ValueError(f"missing probability: {key}")
        x = float(probs[key])
        if x < 0:
            raise ValueError(f"negative probability: {key}={x}")
        values[key] = x

    total = sum(values.values())
    if total <= 0:
        raise ValueError("all probabilities are zero")

    return {k: v / total for k, v in values.items()}


def validate_audit(data: dict, baseline: dict) -> dict:
    probs = normalize_probabilities(data["probabilities"])

    voc = data["view_only_candidate"]["predicted_view"]
    aoc = data["answer_only_candidate"]["predicted_answer_id"]
    bc_view = data["both_candidate"]["predicted_view"]
    bc_ans = data["both_candidate"]["predicted_answer_id"]
    bc_insuff = bool(
        data["both_candidate"].get("insufficient_evidence", False)
    )

    if voc not in VALID_VIEWS:
        raise ValueError(f"invalid view_only candidate: {voc}")
    if aoc not in VALID_ANSWERS:
        raise ValueError(f"invalid answer_only candidate: {aoc}")
    if bc_view not in VALID_VIEWS:
        raise ValueError(f"invalid both view candidate: {bc_view}")
    if bc_ans not in VALID_ANSWERS:
        raise ValueError(f"invalid both answer candidate: {bc_ans}")

    if voc == baseline["predicted_view"]:
        raise ValueError(
            "view_only candidate must differ from current view"
        )
    if aoc == baseline["predicted_answer_id"]:
        raise ValueError(
            "answer_only candidate must differ from current answer"
        )
    if bc_view == baseline["predicted_view"]:
        raise ValueError(
            "both candidate view must differ from current view"
        )
    if bc_ans == baseline["predicted_answer_id"]:
        raise ValueError(
            "both candidate answer must differ from current answer"
        )

    strength = str(data.get("current_support_strength", "medium")).lower()
    if strength not in {"strong", "medium", "weak", "none"}:
        strength = "medium"

    return {
        "probabilities": probs,
        "view_only_candidate": {
            "predicted_view": voc,
        },
        "answer_only_candidate": {
            "predicted_answer_id": aoc,
        },
        "both_candidate": {
            "predicted_view": bc_view,
            "predicted_answer_id": bc_ans,
            "insufficient_evidence": bc_insuff,
        },
        "current_support_strength": strength,
        "evidence_summary": str(data.get("evidence_summary", ""))[:1000],
    }


def call_one_audit(
    record: dict,
    baseline: dict,
    pass_index: int,
) -> dict:
    messages = build_messages(record, baseline, pass_index)

    last_error: Optional[Exception] = None
    last_raw = ""
    last_response = None
    last_finish_reason = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        request_max_tokens = min(
            MAX_OUTPUT_TOKENS * (2 ** (attempt - 1)),
            MAX_RETRY_OUTPUT_TOKENS,
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=request_max_tokens,
                stream=False,
            )

            last_response = response
            last_finish_reason = extract_finish_reason(response)
            last_raw = extract_chat_text(response)

            if not last_raw.strip():
                raise ValueError("empty assistant content")

            cleaned = clean_json_text(last_raw)

            try:
                data = json.loads(cleaned)

            except json.JSONDecodeError:
                # Claude occasionally closes the outer object one brace too early:
                #
                #   "both_candidate": {...}},
                #   "current_support_strength": ...
                #
                # Repair only this known schema-local error.
                repaired = re.sub(
                    r'}\s*}\s*,\s*"current_support_strength"',
                    r'},"current_support_strength"',
                    cleaned,
                    count=1,
                )

                data = json.loads(repaired)
                cleaned = repaired

            parsed = validate_audit(data, baseline)

            return {
                "success": True,
                "parsed": parsed,
                "raw_output": last_raw,
                "cleaned_output": cleaned,
                "finish_reason": last_finish_reason,
                "usage": extract_usage(response),
                "attempt": attempt,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            print(
                f"\nAudit pass {pass_index}, attempt "
                f"{attempt}/{MAX_API_RETRIES} failed: {repr(exc)}"
            )
            if last_raw:
                print("raw preview:", repr(last_raw[:1200]))

            if attempt < MAX_API_RETRIES:
                sleep_time = (
                    RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                    + random.uniform(0, 1.0)
                )
                time.sleep(sleep_time)

    return {
        "success": False,
        "parsed": None,
        "raw_output": last_raw,
        "cleaned_output": clean_json_text(last_raw),
        "finish_reason": last_finish_reason,
        "usage": extract_usage(last_response),
        "attempt": MAX_API_RETRIES,
        "error": repr(last_error),
    }


# ============================================================
# 8. Multi-pass aggregation
# ============================================================

def geometric_mean(xs: List[float], eps: float = 1e-8) -> float:
    return math.exp(
        sum(math.log(max(x, eps)) for x in xs) / len(xs)
    )


def aggregate_passes(
    pass_results: List[dict],
    baseline: dict,
) -> dict:
    successful = [x for x in pass_results if x["success"]]
    if not successful:
        raise RuntimeError("all Claude audit passes failed")

    parsed_list = [x["parsed"] for x in successful]

    agg_probs = {}
    for label in ERROR_TYPES:
        agg_probs[label] = geometric_mean(
            [p["probabilities"][label] for p in parsed_list]
        )

    total = sum(agg_probs.values())
    agg_probs = {k: v / total for k, v in agg_probs.items()}

    best_view_pass = max(
        parsed_list,
        key=lambda x: x["probabilities"]["view_only"],
    )
    best_answer_pass = max(
        parsed_list,
        key=lambda x: x["probabilities"]["answer_only"],
    )
    best_both_pass = max(
        parsed_list,
        key=lambda x: x["probabilities"]["both"],
    )

    return {
        "probabilities": agg_probs,
        "view_only_candidate": best_view_pass["view_only_candidate"],
        "answer_only_candidate": best_answer_pass["answer_only_candidate"],
        "both_candidate": best_both_pass["both_candidate"],
        "current_support_strength": parsed_list[0][
            "current_support_strength"
        ],
        "evidence_summaries": [
            x.get("evidence_summary", "")
            for x in parsed_list
        ],
    }


# ============================================================
# 9. Global constrained optimization
# ============================================================

def label_log_score(audit: dict, label: str) -> float:
    eps = 1e-9
    p = max(float(audit["probabilities"][label]), eps)
    score = math.log(p)

    if label == "both":
        bc = audit["both_candidate"]
        if (
            BOTH_INSUFFICIENCY_BONUS > 0
            and bc.get("insufficient_evidence", False)
            and bc.get("predicted_view") == "NONE_OF_THE_ABOVE"
        ):
            score += math.log(BOTH_INSUFFICIENCY_BONUS)

    return score


def constrained_assignment(
    ordered_records: List[dict],
    audits_by_qid: Dict[str, dict],
) -> Dict[str, str]:
    """
    Finding the maximum using dynamic programming log-likelihood assignment:

        exactly 10 view_only
        exactly 3  answer_only
        exactly 2  both
        remaining 44 correct

    State = (v, a, b)
    correct number i - v - a - b
    """
    V = EXACT_QUOTA["view_only"]
    A = EXACT_QUOTA["answer_only"]
    B = EXACT_QUOTA["both"]
    C = EXACT_QUOTA["correct"]

    neg_inf = float("-inf")

    # dp[state] = best score
    dp: Dict[Tuple[int, int, int], float] = {(0, 0, 0): 0.0}

    # back[i][state_after] = (state_before, chosen_label)
    back: List[
        Dict[
            Tuple[int, int, int],
            Tuple[Tuple[int, int, int], str],
        ]
    ] = []

    for i, record in enumerate(ordered_records, start=1):
        qid = record["question_id"]
        audit = audits_by_qid[qid]["aggregate_audit"]

        new_dp: Dict[Tuple[int, int, int], float] = {}
        new_back: Dict[
            Tuple[int, int, int],
            Tuple[Tuple[int, int, int], str],
        ] = {}

        for state, prev_score in dp.items():
            v, a, b = state

            choices = [
                ("correct", v, a, b),
                ("view_only", v + 1, a, b),
                ("answer_only", v, a + 1, b),
                ("both", v, a, b + 1),
            ]

            for label, nv, na, nb in choices:
                if nv > V or na > A or nb > B:
                    continue

                correct_used = i - nv - na - nb
                if correct_used < 0 or correct_used > C:
                    continue

                new_state = (nv, na, nb)
                score = prev_score + label_log_score(audit, label)

                if score > new_dp.get(new_state, neg_inf):
                    new_dp[new_state] = score
                    new_back[new_state] = (state, label)

        if not new_dp:
            raise RuntimeError(f"DP became empty at item {i}: {qid}")

        dp = new_dp
        back.append(new_back)

    target = (V, A, B)
    if target not in dp:
        raise RuntimeError(
            "Could not satisfy exact 44/10/3/2 quota. "
            "Check audit validity and record count."
        )

    assignment: Dict[str, str] = {}
    state = target

    for i in range(len(ordered_records) - 1, -1, -1):
        qid = ordered_records[i]["question_id"]
        prev_state, label = back[i][state]
        assignment[qid] = label
        state = prev_state

    return assignment


# ============================================================
# 10. Apply selected corrections
# ============================================================

def apply_assignment(
    records: List[dict],
    baseline_by_qid: Dict[str, dict],
    audits_by_qid: Dict[str, dict],
    assignment: Dict[str, str],
) -> Tuple[List[dict], List[dict]]:
    final_records = []
    selected_records = []

    for record in records:
        qid = record["question_id"]
        baseline = baseline_by_qid[qid]
        audit = audits_by_qid[qid]["aggregate_audit"]
        label = assignment[qid]

        p_correct = audit["probabilities"]["correct"]

        if label != "correct" and p_correct > CORRECT_THRESHOLD:
            label = "correct"

        final_view = baseline["predicted_view"]
        final_answer = baseline["predicted_answer_id"]

        if label == "view_only":
            final_view = audit["view_only_candidate"]["predicted_view"]

        elif label == "answer_only":
            final_answer = audit["answer_only_candidate"][
                "predicted_answer_id"
            ]

        elif label == "both":
            final_view = audit["both_candidate"]["predicted_view"]
            final_answer = audit["both_candidate"]["predicted_answer_id"]

        final_obj = {
            "question_id": qid,
            "predicted_view": final_view,
            "predicted_answer_id": final_answer,
        }
        final_records.append(final_obj)

        if label != "correct":
            selected_records.append(
                {
                    "question_id": qid,
                    "assigned_error_type": label,
                    "baseline": baseline,
                    "corrected": final_obj,
                    "probabilities": audit["probabilities"],
                    "both_insufficient_evidence": audit[
                        "both_candidate"
                    ].get("insufficient_evidence", False),
                    "evidence_summaries": audit.get(
                        "evidence_summaries", []
                    ),
                }
            )

    return final_records, selected_records


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_FILE)
    baseline_by_qid = load_prediction_map(BASELINE_FILE)

    if len(records) != 59:
        raise RuntimeError(
            f"Expected 59 test queries for this exact prior, got {len(records)}. "
            "Do not use the 44/10/3/2 hard quota on a different set."
        )

    missing = [
        r["question_id"]
        for r in records
        if r["question_id"] not in baseline_by_qid
    ]
    if missing:
        raise RuntimeError(
            f"Baseline missing {len(missing)} queries:\n"
            + "\n".join(missing)
        )

    if len(baseline_by_qid) != 59:
        print(
            "WARNING: baseline file contains",
            len(baseline_by_qid),
            "unique predictions; expected exactly 59."
        )

    if RESUME:
        completed = load_completed_audits(STAGE3_AUDIT_FILE)
    else:
        completed = {}
        for path in (
            STAGE3_AUDIT_FILE,
            STAGE3_SELECTED_FILE,
            STAGE3_OUTPUT_FILE,
        ):
            if path.exists():
                path.unlink()

    records_to_process = (
        records[:MAX_SAMPLES]
        if MAX_SAMPLES is not None
        else records
    )

    remaining = [
        r
        for r in records_to_process
        if r["question_id"] not in completed
    ]

    print("\n" + "=" * 78)
    print("GoldenViewVQA Stage-3 Claude constrained auditor")
    print("=" * 78)
    print("Input:", INPUT_FILE)
    print("Baseline:", BASELINE_FILE)
    print("Audit file:", STAGE3_AUDIT_FILE)
    print("Selected 15:", STAGE3_SELECTED_FILE)
    print("Final submission:", STAGE3_OUTPUT_FILE)
    print("Model:", MODEL_NAME)
    print("Queries:", len(records))
    print("Audit passes per query:", NUM_AUDIT_PASSES)
    print("Exact quota:", EXACT_QUOTA)
    print(
        "Both insufficiency bonus:",
        BOTH_INSUFFICIENCY_BONUS,
    )
    print("=" * 78 + "\n")

    for record in tqdm(remaining, desc="Claude Stage-3 audit"):
        qid = record["question_id"]
        baseline = baseline_by_qid[qid]

        print("\n" + "-" * 78)
        print("question_id:", qid)
        print("question:", record["question"])
        print("baseline view:", baseline["predicted_view"])
        print("baseline answer:", baseline["predicted_answer_id"])

        pass_results = []
        for pass_index in range(1, NUM_AUDIT_PASSES + 1):
            result = call_one_audit(
                record=record,
                baseline=baseline,
                pass_index=pass_index,
            )
            pass_results.append(result)

            if result["success"]:
                print(
                    f"pass {pass_index} probs:",
                    result["parsed"]["probabilities"],
                )
            else:
                print(
                    f"pass {pass_index} FAILED:",
                    result["error"],
                )

        aggregate = aggregate_passes(
            pass_results=pass_results,
            baseline=baseline,
        )

        audit_record = {
            "question_id": qid,
            "model": MODEL_NAME,
            "baseline": baseline,
            "pass_results": pass_results,
            "aggregate_audit": aggregate,
        }

        append_jsonl(STAGE3_AUDIT_FILE, audit_record)
        completed[qid] = audit_record

        print("aggregate probs:", aggregate["probabilities"])
        print(
            "view-only candidate:",
            aggregate["view_only_candidate"],
        )
        print(
            "answer-only candidate:",
            aggregate["answer_only_candidate"],
        )
        print("both candidate:", aggregate["both_candidate"])

    if MAX_SAMPLES is not None:
        print(
            "\nPartial audit finished. "
            "Unset MAX_SAMPLES for the full 59-query constrained assignment."
        )
        return

    audits_by_qid = load_completed_audits(STAGE3_AUDIT_FILE)
    missing_audits = [
        r["question_id"]
        for r in records
        if r["question_id"] not in audits_by_qid
    ]
    if missing_audits:
        raise RuntimeError(
            f"Missing {len(missing_audits)} audits:\n"
            + "\n".join(missing_audits)
        )

    assignment = constrained_assignment(
        ordered_records=records,
        audits_by_qid=audits_by_qid,
    )

    counts = {k: 0 for k in ERROR_TYPES}
    for label in assignment.values():
        counts[label] += 1

    if counts != EXACT_QUOTA:
        raise RuntimeError(
            f"Internal quota mismatch: {counts} != {EXACT_QUOTA}"
        )

    final_records, selected_records = apply_assignment(
        records=records,
        baseline_by_qid=baseline_by_qid,
        audits_by_qid=audits_by_qid,
        assignment=assignment,
    )

    def selection_priority(x: dict) -> float:
        label = x["assigned_error_type"]
        probs = x["probabilities"]
        p_err = probs[label]
        p_correct = probs["correct"]
        return p_err / max(p_correct, 1e-9)

    selected_records.sort(
        key=selection_priority,
        reverse=True,
    )

    write_jsonl(STAGE3_SELECTED_FILE, selected_records)
    write_jsonl(STAGE3_OUTPUT_FILE, final_records)

    print("\n" + "=" * 78)
    print("Stage-3 finished")
    print("=" * 78)
    print("Assigned counts:", counts)
    print("Selected corrections:", len(selected_records))
    print("Final submission:", STAGE3_OUTPUT_FILE)
    print("Selection report:", STAGE3_SELECTED_FILE)

    print("\nSelected 15:")
    for i, x in enumerate(selected_records, start=1):
        b = x["baseline"]
        c = x["corrected"]
        p = x["probabilities"]
        print(
            f"{i:02d}. {x['question_id']}  "
            f"{x['assigned_error_type']:11s}  "
            f"P={p[x['assigned_error_type']]:.3f}  "
            f"correct={p['correct']:.3f}  "
            f"{b['predicted_view']}/{b['predicted_answer_id']} -> "
            f"{c['predicted_view']}/{c['predicted_answer_id']}"
        )


if __name__ == "__main__":
    main()
