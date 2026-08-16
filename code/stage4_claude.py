#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoldenViewVQA Stage-4: Claude group-level prior-guided verification

"""

import base64
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
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

NUM_VERIFY_PASSES = int(os.getenv("NUM_VERIFY_PASSES", "2"))


CORRECT_THRESHOLD = float(os.getenv("CORRECT_THRESHOLD", "0.5"))

# Only groups with >= 2 questions contain cross-question information.
MIN_GROUP_SIZE = int(os.getenv("MIN_GROUP_SIZE", "2"))

RESUME = os.getenv("RESUME", "true").lower() == "true"
REGENERATE_GROUP_PRIORS = (
    os.getenv("REGENERATE_GROUP_PRIORS", "false").lower() == "true"
)


MAX_SAMPLES_ENV = os.getenv("MAX_SAMPLES", "")
MAX_SAMPLES = int(MAX_SAMPLES_ENV) if MAX_SAMPLES_ENV else None

SEND_IMAGE_DETAIL = os.getenv("SEND_IMAGE_DETAIL", "false").lower() == "true"
IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "auto").lower()


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

STAGE3_FILE = Path(
    os.getenv(
        "STAGE3_FILE",
        str(OUTPUT_DIR / "claude_stage3_final_test.jsonl"),
    )
)

STAGE4_GROUP_PRIOR_FILE = Path(
    os.getenv(
        "STAGE4_GROUP_PRIOR_FILE",
        str(OUTPUT_DIR / "claude_stage4_group_priors.jsonl"),
    )
)

STAGE4_AUDIT_FILE = Path(
    os.getenv(
        "STAGE4_AUDIT_FILE",
        str(OUTPUT_DIR / "claude_stage4_audit.jsonl"),
    )
)

STAGE4_SELECTED_FILE = Path(
    os.getenv(
        "STAGE4_SELECTED_FILE",
        str(OUTPUT_DIR / "claude_stage4_selected.jsonl"),
    )
)

STAGE4_OUTPUT_FILE = Path(
    os.getenv(
        "STAGE4_OUTPUT_FILE",
        str(OUTPUT_DIR / "claude_stage4_final_test.jsonl"),
    )
)


# ============================================================
# 2. Labels
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

QUESTION_SUFFIX_RE = re.compile(
    r"_(causality|counterfactual|intent_prediction)$"
)


# ============================================================
# 3. Client
# ============================================================

def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export your API key before running."
        )

    print("Using base URL:", OPENAI_BASE_URL)
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
            raise ValueError(f"{qid}: invalid predicted view: {view}")
        if ans not in VALID_ANSWERS:
            raise ValueError(f"{qid}: invalid predicted answer: {ans}")

        result[qid] = {
            "question_id": qid,
            "predicted_view": view,
            "predicted_answer_id": ans,
        }
    return result


def load_jsonl_map(path: Path, key: str) -> Dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for obj in read_jsonl(path):
        value = obj.get(key)
        if value is not None:
            result[value] = obj
    return result


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


def api_json_request(messages: List[dict]) -> dict:
    """Call the OpenAI-compatible endpoint and parse one JSON object."""
    last_error: Optional[Exception] = None
    last_raw = ""
    last_response = None

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
            last_raw = extract_chat_text(response)
            if not last_raw.strip():
                raise ValueError("empty assistant content")

            cleaned = clean_json_text(last_raw)
            data = json.loads(cleaned)

            return {
                "success": True,
                "data": data,
                "raw_output": last_raw,
                "cleaned_output": cleaned,
                "finish_reason": extract_finish_reason(response),
                "usage": extract_usage(response),
                "attempt": attempt,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            print(
                f"\nAPI attempt {attempt}/{MAX_API_RETRIES} failed: {repr(exc)}"
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
        "data": None,
        "raw_output": last_raw,
        "cleaned_output": clean_json_text(last_raw),
        "finish_reason": extract_finish_reason(last_response),
        "usage": extract_usage(last_response),
        "attempt": MAX_API_RETRIES,
        "error": repr(last_error),
    }


# ============================================================
# 5. Grouping and scene checks
# ============================================================

def base_group_id(question_id: str) -> str:
    """Remove only the known reasoning-type suffix from question_id."""
    base = QUESTION_SUFFIX_RE.sub("", question_id)
    if base == question_id:
        raise ValueError(
            f"Cannot infer base group id from question_id: {question_id}"
        )
    return base


def build_groups(records: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[base_group_id(record["question_id"])].append(record)
    return dict(groups)


def scene_signature(record: dict) -> Tuple[Tuple[str, str], ...]:
    return tuple((cam, str(record["views"].get(cam, ""))) for cam in CAMERA_ORDER)


def validate_group_scenes(groups: Dict[str, List[dict]]) -> None:
    for gid, members in groups.items():
        if len(members) < 2:
            continue

        signatures = {scene_signature(x) for x in members}
        if len(signatures) != 1:
            member_ids = [x["question_id"] for x in members]
            raise RuntimeError(
                f"Group {gid} does not share exactly the same six-view scene: "
                f"{member_ids}"
            )


# ============================================================
# 6. Images
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
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_types[suffix]};base64,{encoded}"


# ============================================================
# 7. Group-prior generation
# ============================================================

GROUP_PRIOR_SYSTEM = """
You construct a soft cross-question prior for GoldenViewVQA.

The provided question stems belong to the SAME six-view driving scene. They
share a base query ID but may ask different reasoning types (causality,
counterfactual, intent prediction).

Your job is NOT to solve any individual question and NOT to invent scene facts.
Question wording is only a semantic hint about what may be relevant. Extract
recurring entities, spatial relations, action constraints, and uncertainty
checks that could help a later visual verifier inspect the six images.

Important rules:
- Treat every inferred item as a hypothesis to verify visually, not ground truth.
- Do not infer a golden/supporting camera from the question ID.
- Do not use or assume hidden labels.
- Keep the prior concise and useful for later verification.
- Return exactly one JSON object, with no markdown or extra prose.
""".strip()


def build_group_prior_prompt(group_id: str, members: List[dict]) -> str:
    lines = []
    for idx, record in enumerate(members, start=1):
        lines.append(
            f"{idx}. question_id={record['question_id']}\n"
            f"   reasoning_type={record.get('question_group', '')}\n"
            f"   stem={record['question']}"
        )

    stems = "\n\n".join(lines)

    return f"""
Group ID: {group_id}
Number of questions: {len(members)}

QUESTION STEMS FROM THE SAME SIX-VIEW SCENE:
{stems}

Build a group-level prior that a later verifier can use only as auxiliary
context while inspecting the actual six images.

Required JSON schema:
{{
  "scene_focus": "one concise sentence describing the likely shared semantic focus",
  "recurring_entities": ["entity or road-user hypotheses"],
  "spatial_relations_to_check": ["specific relations that should be verified visually"],
  "driving_constraints_to_check": ["possible scene/action constraints"],
  "cross_question_consistency_cues": ["how the stems constrain one another without assuming they are true"],
  "uncertainty_checks": ["conditions under which evidence may be insufficient"]
}}
""".strip()


def validate_group_prior(data: dict) -> dict:
    def string_list(name: str) -> List[str]:
        value = data.get(name, [])
        if not isinstance(value, list):
            value = [str(value)] if value else []
        return [str(x)[:300] for x in value[:12]]

    return {
        "scene_focus": str(data.get("scene_focus", ""))[:500],
        "recurring_entities": string_list("recurring_entities"),
        "spatial_relations_to_check": string_list("spatial_relations_to_check"),
        "driving_constraints_to_check": string_list("driving_constraints_to_check"),
        "cross_question_consistency_cues": string_list(
            "cross_question_consistency_cues"
        ),
        "uncertainty_checks": string_list("uncertainty_checks"),
    }


def generate_group_prior(group_id: str, members: List[dict]) -> dict:
    messages = [
        {"role": "system", "content": GROUP_PRIOR_SYSTEM},
        {
            "role": "user",
            "content": build_group_prior_prompt(group_id, members),
        },
    ]

    result = api_json_request(messages)
    if not result["success"]:
        raise RuntimeError(
            f"Failed to generate group prior for {group_id}: {result['error']}"
        )

    prior = validate_group_prior(result["data"])
    return {
        "group_id": group_id,
        "member_question_ids": [x["question_id"] for x in members],
        "group_size": len(members),
        "prior": prior,
        "raw_output": result["raw_output"],
        "usage": result["usage"],
    }


# ============================================================
# 8. Stage-4 per-instance verification
# ============================================================

VERIFY_SYSTEM = """
You are the Stage-4 group-level verifier for GoldenViewVQA.

You receive a strong current Stage-3 answer-view prediction, six synchronized
camera views, the target question and answer options, and a GROUP-LEVEL PRIOR
constructed only from question stems that share the same six-view scene.

The group prior is a SOFT HYPOTHESIS, not ground truth. It may be incomplete or
misleading. The target question and the actual images always dominate.

Golden-view rule:
Choose the single camera containing the decisive visual evidence needed to
justify the answer. If none of the six cameras provides sufficient direct
evidence, use NONE_OF_THE_ABOVE.

Be conservative. Do not change a Stage-3 prediction merely because another
answer or view is plausible. Recommend a change only when the visual evidence,
target question, and useful cross-question constraints provide stronger support
for a different answer-view pair.

Return exactly one JSON object, with no markdown or prose outside JSON.
""".strip()


def build_verify_prompt(
    record: dict,
    stage3: dict,
    group_id: str,
    group_size: int,
    group_prior: dict,
    pass_index: int,
) -> str:
    options = record["options"]
    current_ans = stage3["predicted_answer_id"]

    return f"""
Verification pass: {pass_index}

Group ID: {group_id}
Group size: {group_size}

GROUP-LEVEL PRIOR (soft auxiliary context):
{json.dumps(group_prior, ensure_ascii=False, indent=2)}

TARGET QUESTION:
Question ID: {record['question_id']}
Reasoning type: {record.get('question_group', '')}
Question: {record['question']}

Answer options:
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}

CURRENT STAGE-3 PREDICTION:
- View: {stage3['predicted_view']}
- Answer ID: {current_ans}
- Answer text: {options[current_ans]}

Inspect all six images. First decide whether the current Stage-3 answer-view
pair is jointly correct. Then give the best final answer-view pair.

Required JSON schema:
{{
  "current_joint_correct_probability": 80,
  "recommended_view": "CAM_FRONT",
  "recommended_answer_id": "A",
  "group_prior_effect": "supports|neutral|conflicts",
  "visual_support_strength": "strong|medium|weak|none",
  "evidence_summary": "brief concrete explanation, <= 80 words"
}}

The probability may be given on either a 0-1 or 0-100 scale.
""".strip()


def build_verify_messages(
    record: dict,
    stage3: dict,
    group_id: str,
    group_size: int,
    group_prior: dict,
    pass_index: int,
) -> List[dict]:
    content: List[dict] = [
        {
            "type": "text",
            "text": build_verify_prompt(
                record=record,
                stage3=stage3,
                group_id=group_id,
                group_size=group_size,
                group_prior=group_prior,
                pass_index=pass_index,
            ),
        }
    ]

    for camera in CAMERA_ORDER:
        if camera not in record["views"]:
            raise KeyError(f"{record['question_id']} missing camera: {camera}")

        image_path = (NUSCENES_ROOT / record["views"][camera]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for {record['question_id']} / {camera}: {image_path}"
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
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": content},
    ]


def normalize_probability(x: Any) -> float:
    p = float(x)
    if p < 0:
        raise ValueError(f"negative probability: {p}")
    if p > 1.0:
        p = p / 100.0
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range after normalization: {p}")
    return p


def validate_verify_result(data: dict) -> dict:
    p_correct = normalize_probability(data["current_joint_correct_probability"])
    view = data["recommended_view"]
    ans = data["recommended_answer_id"]

    if view not in VALID_VIEWS:
        raise ValueError(f"invalid recommended_view: {view}")
    if ans not in VALID_ANSWERS:
        raise ValueError(f"invalid recommended_answer_id: {ans}")

    prior_effect = str(data.get("group_prior_effect", "neutral")).lower()
    if prior_effect not in {"supports", "neutral", "conflicts"}:
        prior_effect = "neutral"

    strength = str(data.get("visual_support_strength", "medium")).lower()
    if strength not in {"strong", "medium", "weak", "none"}:
        strength = "medium"

    return {
        "current_joint_correct_probability": p_correct,
        "recommended_view": view,
        "recommended_answer_id": ans,
        "group_prior_effect": prior_effect,
        "visual_support_strength": strength,
        "evidence_summary": str(data.get("evidence_summary", ""))[:1200],
    }


def call_verify_pass(
    record: dict,
    stage3: dict,
    group_id: str,
    group_size: int,
    group_prior: dict,
    pass_index: int,
) -> dict:
    messages = build_verify_messages(
        record=record,
        stage3=stage3,
        group_id=group_id,
        group_size=group_size,
        group_prior=group_prior,
        pass_index=pass_index,
    )

    result = api_json_request(messages)
    if not result["success"]:
        return result

    try:
        parsed = validate_verify_result(result["data"])
        result["parsed"] = parsed
        return result
    except Exception as exc:
        return {
            **result,
            "success": False,
            "parsed": None,
            "error": repr(exc),
        }


# ============================================================
# 9. Multi-pass aggregation
# ============================================================

def geometric_mean(xs: List[float], eps: float = 1e-8) -> float:
    return math.exp(sum(math.log(max(x, eps)) for x in xs) / len(xs))


def aggregate_verify_passes(pass_results: List[dict], stage3: dict) -> dict:
    successful = [x for x in pass_results if x.get("success") and x.get("parsed")]
    if not successful:
        return {
            "success": False,
            "current_joint_correct_probability": 1.0,
            "recommended_view": stage3["predicted_view"],
            "recommended_answer_id": stage3["predicted_answer_id"],
            "group_prior_effect": "neutral",
            "visual_support_strength": "medium",
            "evidence_summaries": [],
        }

    parsed = [x["parsed"] for x in successful]

    # Aggregate p(current_correct) conservatively with geometric mean.
    p_correct = geometric_mean(
        [x["current_joint_correct_probability"] for x in parsed]
    )

    # Candidate pair voting weighted by confidence that the current pair is wrong.
    pair_score: Dict[Tuple[str, str], float] = defaultdict(float)
    pair_count: Counter = Counter()
    for x in parsed:
        pair = (x["recommended_view"], x["recommended_answer_id"])
        weight = max(1.0 - x["current_joint_correct_probability"], 1e-6)
        pair_score[pair] += weight
        pair_count[pair] += 1

    best_pair = max(
        pair_score,
        key=lambda pair: (pair_score[pair], pair_count[pair]),
    )

    # Majority/weighted summary labels.
    prior_effect = Counter(x["group_prior_effect"] for x in parsed).most_common(1)[0][0]
    visual_strength = Counter(
        x["visual_support_strength"] for x in parsed
    ).most_common(1)[0][0]

    return {
        "success": True,
        "current_joint_correct_probability": p_correct,
        "recommended_view": best_pair[0],
        "recommended_answer_id": best_pair[1],
        "group_prior_effect": prior_effect,
        "visual_support_strength": visual_strength,
        "evidence_summaries": [x["evidence_summary"] for x in parsed],
        "num_successful_passes": len(successful),
    }


# ============================================================
# 10. Apply Stage-4 confidence-gated corrections
# ============================================================

def apply_stage4(
    records: List[dict],
    stage3_by_qid: Dict[str, dict],
    audits_by_qid: Dict[str, dict],
    groups: Dict[str, List[dict]],
) -> Tuple[List[dict], List[dict]]:
    final_records: List[dict] = []
    selected: List[dict] = []

    group_sizes = {gid: len(members) for gid, members in groups.items()}

    for record in records:
        qid = record["question_id"]
        gid = base_group_id(qid)
        stage3 = stage3_by_qid[qid]

        final_view = stage3["predicted_view"]
        final_answer = stage3["predicted_answer_id"]
        changed = False
        reason = "keep"

        audit_record = audits_by_qid.get(qid)

        if group_sizes[gid] < MIN_GROUP_SIZE:
            reason = "singleton_group"

        elif not audit_record or not audit_record.get("aggregate_audit"):
            reason = "missing_or_failed_audit"

        else:
            audit = audit_record["aggregate_audit"]
            p_correct = float(audit["current_joint_correct_probability"])
            rec_view = audit["recommended_view"]
            rec_answer = audit["recommended_answer_id"]

            if p_correct <= CORRECT_THRESHOLD and (
                rec_view != final_view or rec_answer != final_answer
            ):
                final_view = rec_view
                final_answer = rec_answer
                changed = True
                reason = "confidence_gated_change"
            else:
                reason = "confidence_gate_keep"

        final_obj = {
            "question_id": qid,
            "predicted_view": final_view,
            "predicted_answer_id": final_answer,
        }
        final_records.append(final_obj)

        if changed:
            audit = audit_record["aggregate_audit"]
            selected.append(
                {
                    "question_id": qid,
                    "group_id": gid,
                    "group_size": group_sizes[gid],
                    "stage3": stage3,
                    "stage4": final_obj,
                    "current_joint_correct_probability": audit[
                        "current_joint_correct_probability"
                    ],
                    "group_prior_effect": audit["group_prior_effect"],
                    "visual_support_strength": audit["visual_support_strength"],
                    "evidence_summaries": audit.get("evidence_summaries", []),
                    "reason": reason,
                }
            )

    return final_records, selected


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_FILE)
    stage3_by_qid = load_prediction_map(STAGE3_FILE)

    missing = [
        r["question_id"]
        for r in records
        if r["question_id"] not in stage3_by_qid
    ]
    if missing:
        raise RuntimeError(
            f"Stage-3 file is missing {len(missing)} questions:\n"
            + "\n".join(missing)
        )

    groups = build_groups(records)
    validate_group_scenes(groups)

    group_size_hist = Counter(len(v) for v in groups.values())

    print("\n" + "=" * 78)
    print("GoldenViewVQA Stage-4 Claude group-prior verifier")
    print("=" * 78)
    print("Input:", INPUT_FILE)
    print("Stage-3 baseline:", STAGE3_FILE)
    print("Group prior file:", STAGE4_GROUP_PRIOR_FILE)
    print("Audit file:", STAGE4_AUDIT_FILE)
    print("Selected corrections:", STAGE4_SELECTED_FILE)
    print("Final Stage-4 submission:", STAGE4_OUTPUT_FILE)
    print("Model:", MODEL_NAME)
    print("Questions:", len(records))
    print("Groups:", len(groups))
    print("Group size histogram:", dict(sorted(group_size_hist.items())))
    print("Minimum group size:", MIN_GROUP_SIZE)
    print("Correct threshold:", CORRECT_THRESHOLD)
    print("Verification passes:", NUM_VERIFY_PASSES)
    print("=" * 78 + "\n")

    # --------------------------------------------------------
    # A. Build/reuse one semantic prior per same-scene group.
    # --------------------------------------------------------
    if REGENERATE_GROUP_PRIORS and STAGE4_GROUP_PRIOR_FILE.exists():
        STAGE4_GROUP_PRIOR_FILE.unlink()

    existing_priors = load_jsonl_map(STAGE4_GROUP_PRIOR_FILE, "group_id")

    for gid, members in tqdm(groups.items(), desc="Stage-4 group priors"):
        if len(members) < MIN_GROUP_SIZE:
            continue
        if gid in existing_priors and not REGENERATE_GROUP_PRIORS:
            continue

        prior_record = generate_group_prior(gid, members)
        append_jsonl(STAGE4_GROUP_PRIOR_FILE, prior_record)
        existing_priors[gid] = prior_record

        print("\nGroup:", gid)
        print("Members:", prior_record["member_question_ids"])
        print("Prior focus:", prior_record["prior"]["scene_focus"])

    # Check all non-singleton groups have priors.
    missing_priors = [
        gid
        for gid, members in groups.items()
        if len(members) >= MIN_GROUP_SIZE and gid not in existing_priors
    ]
    if missing_priors:
        raise RuntimeError(
            "Missing group priors for: " + ", ".join(missing_priors)
        )

    # --------------------------------------------------------
    # B. Per-instance Claude verification with visual evidence
    #    + the group prior.
    # --------------------------------------------------------
    if not RESUME and STAGE4_AUDIT_FILE.exists():
        STAGE4_AUDIT_FILE.unlink()

    completed_audits = load_jsonl_map(STAGE4_AUDIT_FILE, "question_id")

    records_to_process = records[:MAX_SAMPLES] if MAX_SAMPLES else records

    for record in tqdm(records_to_process, desc="Stage-4 verification"):
        qid = record["question_id"]
        gid = base_group_id(qid)
        members = groups[gid]

        if len(members) < MIN_GROUP_SIZE:
            continue
        if qid in completed_audits and RESUME:
            continue

        stage3 = stage3_by_qid[qid]
        group_prior = existing_priors[gid]["prior"]

        print("\n" + "-" * 78)
        print("question_id:", qid)
        print("group_id:", gid)
        print("question:", record["question"])
        print(
            "stage3:",
            stage3["predicted_view"],
            stage3["predicted_answer_id"],
        )

        pass_results = []
        for pass_index in range(1, NUM_VERIFY_PASSES + 1):
            result = call_verify_pass(
                record=record,
                stage3=stage3,
                group_id=gid,
                group_size=len(members),
                group_prior=group_prior,
                pass_index=pass_index,
            )
            pass_results.append(result)

            if result.get("success") and result.get("parsed"):
                p = result["parsed"]
                print(
                    f"pass {pass_index}: "
                    f"correct={p['current_joint_correct_probability']:.3f}, "
                    f"recommend={p['recommended_view']}/{p['recommended_answer_id']}, "
                    f"prior={p['group_prior_effect']}"
                )
            else:
                print(f"pass {pass_index} FAILED: {result.get('error')}")

        aggregate = aggregate_verify_passes(pass_results, stage3)

        audit_record = {
            "question_id": qid,
            "group_id": gid,
            "group_size": len(members),
            "model": MODEL_NAME,
            "stage3": stage3,
            "group_prior": group_prior,
            "pass_results": pass_results,
            "aggregate_audit": aggregate,
        }

        append_jsonl(STAGE4_AUDIT_FILE, audit_record)
        completed_audits[qid] = audit_record

        print("aggregate:", aggregate)

    if MAX_SAMPLES is not None:
        print(
            "\nPartial Stage-4 audit finished. Unset MAX_SAMPLES to generate "
            "the full submission."
        )
        return

    # Reload from disk for a clean final application.
    audits_by_qid = load_jsonl_map(STAGE4_AUDIT_FILE, "question_id")

    final_records, selected = apply_stage4(
        records=records,
        stage3_by_qid=stage3_by_qid,
        audits_by_qid=audits_by_qid,
        groups=groups,
    )

    selected.sort(
        key=lambda x: x["current_joint_correct_probability"]
    )

    write_jsonl(STAGE4_SELECTED_FILE, selected)
    write_jsonl(STAGE4_OUTPUT_FILE, final_records)

    # Summary of answer/view change types.
    change_counts = Counter()
    for x in selected:
        old = x["stage3"]
        new = x["stage4"]
        view_changed = old["predicted_view"] != new["predicted_view"]
        ans_changed = old["predicted_answer_id"] != new["predicted_answer_id"]
        if view_changed and ans_changed:
            change_counts["both"] += 1
        elif view_changed:
            change_counts["view_only"] += 1
        elif ans_changed:
            change_counts["answer_only"] += 1

    print("\n" + "=" * 78)
    print("Stage-4 finished")
    print("=" * 78)
    print("Applied corrections:", len(selected))
    print("Change types:", dict(change_counts))
    print("Final output:", STAGE4_OUTPUT_FILE)
    print("Selected report:", STAGE4_SELECTED_FILE)

    for i, x in enumerate(selected, start=1):
        old = x["stage3"]
        new = x["stage4"]
        print(
            f"{i:02d}. {x['question_id']}  "
            f"p_correct={x['current_joint_correct_probability']:.3f}  "
            f"{old['predicted_view']}/{old['predicted_answer_id']} -> "
            f"{new['predicted_view']}/{new['predicted_answer_id']}"
        )


if __name__ == "__main__":
    main()
