#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GoldenViewVQA Stage 4: cross-split group-level GPT-5.6 verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


# ============================================================
# 0. Runtime configuration
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.6")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high").lower()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "360"))
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", "4"))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "3.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
MAX_RETRY_OUTPUT_TOKENS = int(os.getenv("MAX_RETRY_OUTPUT_TOKENS", "8192"))

IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "auto").lower()
MIN_GROUP_SIZE = int(os.getenv("MIN_GROUP_SIZE", "2"))
RESUME = os.getenv("RESUME", "true").lower() == "true"
REGENERATE_SEMANTIC_GROUPS = (
    os.getenv("REGENERATE_SEMANTIC_GROUPS", "false").lower() == "true"
)
REGENERATE_GROUP_AUDITS = (
    os.getenv("REGENERATE_GROUP_AUDITS", "false").lower() == "true"
)

# MAX_SAMPLES remains accepted as a compatibility alias for earlier scripts.
MAX_GROUPS_ENV = os.getenv("MAX_GROUPS", os.getenv("MAX_SAMPLES", "")).strip()
MAX_GROUPS = int(MAX_GROUPS_ENV) if MAX_GROUPS_ENV else None

SEMANTIC_FILTER_VERSION = "cross_split_text_filter_v1"
GROUP_VERIFY_VERSION = "cross_split_joint_verify_v1"


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
DATA_DIR = GVQA_ROOT / "data"

VAL_INPUT_FILE = Path(
    os.getenv("VAL_INPUT_FILE", str(DATA_DIR / "eval_inputs.jsonl"))
)
VAL_LABEL_FILE = Path(
    os.getenv("VAL_LABEL_FILE", str(DATA_DIR / "eval.jsonl"))
)
TEST_INPUT_FILE = Path(
    os.getenv("TEST_INPUT_FILE", str(DATA_DIR / "test_inputs.jsonl"))
)
NUSCENES_ROOT = Path(
    os.getenv("NUSCENES_ROOT", str(DATA_DIR / "nuscenes"))
)

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs/test"))
)
STAGE3_FILE = Path(
    os.getenv(
        "STAGE3_FILE",
        str(OUTPUT_DIR / "claude_stage3_final_test.jsonl"),
    )
)
STAGE4_SEMANTIC_GROUP_FILE = Path(
    os.getenv(
        "STAGE4_SEMANTIC_GROUP_FILE",
        str(OUTPUT_DIR / "gpt56_stage4_semantic_groups.jsonl"),
    )
)
STAGE4_GROUP_AUDIT_FILE = Path(
    os.getenv(
        "STAGE4_GROUP_AUDIT_FILE",
        str(OUTPUT_DIR / "gpt56_stage4_group_audits.jsonl"),
    )
)
STAGE4_SELECTED_FILE = Path(
    os.getenv(
        "STAGE4_SELECTED_FILE",
        str(OUTPUT_DIR / "gpt56_stage4_selected.jsonl"),
    )
)
STAGE4_OUTPUT_FILE = Path(
    os.getenv(
        "STAGE4_OUTPUT_FILE",
        str(OUTPUT_DIR / "gpt56_stage4_final_test.jsonl"),
    )
)


# ============================================================
# 2. Task labels and structured outputs
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

BASE_SCENE_RE = re.compile(r"^(sfall_\d+)(?:_|$)", re.IGNORECASE)

ViewLabel = Literal[
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "NONE_OF_THE_ABOVE",
]
AnswerLabel = Literal["A", "B", "C", "D"]


class TextualGroupPrior(BaseModel):
    shared_focus: str
    core_entities: List[str]
    local_spatial_relations: List[str]
    semantic_causal_behavioral_links: List[str]
    transfer_cautions: List[str]


class SemanticPartition(BaseModel):
    member_labels: List[str]
    textual_prior: TextualGroupPrior


class SemanticFilterOutput(BaseModel):
    groups: List[SemanticPartition]


class GroupTestPrediction(BaseModel):
    question_label: str
    predicted_view: ViewLabel
    predicted_answer_id: AnswerLabel
    validation_prior_relation: Literal[
        "consistent",
        "question_specific_override",
        "not_applicable",
    ]
    evidence_summary: str


class GroupVerificationOutput(BaseModel):
    group_consistency_summary: str
    predictions: List[GroupTestPrediction]


# ============================================================
# 3. JSONL and normalization helpers
# ============================================================

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            records.append(value)
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_latest_jsonl_map(path: Path, key: str) -> Dict[str, dict]:
    if not path.exists():
        return {}

    result: Dict[str, dict] = {}
    for record in read_jsonl(path):
        value = record.get(key)
        if isinstance(value, str) and value:
            result[value] = record
    return result


def index_by_question_id(records: Sequence[dict], source: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for record in records:
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Record without a valid question_id in {source}")
        if qid in result:
            raise ValueError(f"Duplicate question_id {qid} in {source}")
        result[qid] = record
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def model_to_dict(value: BaseModel) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def normalize_view(value: Any, context: str) -> str:
    aliases = {
        "NONE": "NONE_OF_THE_ABOVE",
        "NONE_OF_ABOVE": "NONE_OF_THE_ABOVE",
        "NONE-OF-THE-ABOVE": "NONE_OF_THE_ABOVE",
    }
    normalized = aliases.get(str(value).strip().upper(), str(value).strip().upper())
    if normalized not in VALID_VIEWS:
        raise ValueError(f"{context}: invalid view label {value!r}")
    return normalized


def normalize_answer(value: Any, context: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in VALID_ANSWERS:
        raise ValueError(f"{context}: invalid answer label {value!r}")
    return normalized


def first_present(records: Sequence[dict], names: Sequence[str]) -> Any:
    for record in records:
        for name in names:
            value = record.get(name)
            if value is not None and str(value).strip():
                return value
    return None


def normalize_observation_path(value: Any) -> str:
    normalized = str(value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


# ============================================================
# 4. OpenAI Responses API
# ============================================================

def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. It is only required when an uncached "
            "semantic filter or group verification call must be made."
        )

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": API_TIMEOUT,
        "max_retries": 0,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    print("Using model:", MODEL_NAME)
    print("Using base URL:", OPENAI_BASE_URL or "OpenAI default")
    return OpenAI(**kwargs)


def extract_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details is not None:
        result["cached_input_tokens"] = getattr(
            input_details,
            "cached_tokens",
            None,
        )
    return result


def call_structured(
    client: OpenAI,
    input_messages: List[dict],
    output_schema: Type[BaseModel],
    operation_name: str,
    validator: Optional[Callable[[dict], dict]] = None,
) -> dict:
    last_error: Optional[Exception] = None
    last_output = ""

    for attempt in range(1, MAX_API_RETRIES + 1):
        last_output = ""
        request_max_tokens = min(
            MAX_OUTPUT_TOKENS * (2 ** (attempt - 1)),
            MAX_RETRY_OUTPUT_TOKENS,
        )
        try:
            response = client.responses.parse(
                model=MODEL_NAME,
                input=input_messages,
                reasoning={"effort": REASONING_EFFORT},
                max_output_tokens=request_max_tokens,
                text_format=output_schema,
            )
            last_output = response.output_text or ""
            parsed_model = response.output_parsed
            if parsed_model is None:
                raise ValueError("response.output_parsed is empty")

            parsed = model_to_dict(parsed_model)
            if validator is not None:
                parsed = validator(parsed)

            return {
                "parsed": parsed,
                "raw_output": last_output,
                "response_id": getattr(response, "id", None),
                "usage": extract_usage(response),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            print(
                f"\n{operation_name} attempt {attempt}/{MAX_API_RETRIES} "
                f"failed: {exc!r}"
            )
            if last_output:
                print("Raw output preview:", repr(last_output[:1200]))
            if attempt < MAX_API_RETRIES:
                delay = (
                    RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                    + random.uniform(0.0, 1.0)
                )
                time.sleep(delay)

    raise RuntimeError(
        f"{operation_name} failed after {MAX_API_RETRIES} attempts"
    ) from last_error


# ============================================================
# 5. Validation/test loading
# ============================================================

def normalize_input_record(record: dict, split: str) -> dict:
    qid = record.get("question_id")
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"{split} record has no valid question_id")

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{qid}: missing question text")

    raw_options = record.get("options")
    if not isinstance(raw_options, dict):
        raise ValueError(f"{qid}: options must be an object")
    options: Dict[str, str] = {}
    for answer_id in sorted(VALID_ANSWERS):
        option = raw_options.get(answer_id)
        if option is None:
            raise ValueError(f"{qid}: missing option {answer_id}")
        options[answer_id] = str(option)

    raw_views = record.get("views")
    if not isinstance(raw_views, dict):
        raise ValueError(f"{qid}: views must be an object")
    views: Dict[str, str] = {}
    for camera in CAMERA_ORDER:
        value = raw_views.get(camera)
        if value is None or not str(value).strip():
            raise ValueError(f"{qid}: missing synchronized view {camera}")
        views[camera] = str(value)

    return {
        "question_id": qid,
        "split": split,
        "question_group": str(record.get("question_group", "")),
        "question": question.strip(),
        "options": options,
        "views": views,
    }


def load_prediction_map(path: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for record in read_jsonl(path):
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Prediction without question_id in {path}")
        if qid in result:
            raise ValueError(f"Duplicate Stage-3 prediction for {qid}")
        result[qid] = {
            "question_id": qid,
            "predicted_view": normalize_view(
                record.get("predicted_view"),
                f"Stage-3 {qid}",
            ),
            "predicted_answer_id": normalize_answer(
                record.get("predicted_answer_id"),
                f"Stage-3 {qid}",
            ),
        }
    return result


def load_validation_members() -> List[dict]:
    input_records = read_jsonl(VAL_INPUT_FILE)
    label_records = read_jsonl(VAL_LABEL_FILE)
    label_by_qid = index_by_question_id(label_records, VAL_LABEL_FILE)

    members: List[dict] = []
    missing_labels: List[str] = []
    for input_record in input_records:
        member = normalize_input_record(input_record, "validation")
        qid = member["question_id"]
        label_record = label_by_qid.get(qid)
        if label_record is None:
            missing_labels.append(qid)
            continue

        view_value = first_present(
            [label_record, input_record],
            [
                "golden_view",
                "gold_view",
                "gt_view",
                "correct_view",
                "supporting_view",
                "answer_view",
                "view",
                "predicted_view",
            ],
        )
        answer_value = first_present(
            [label_record, input_record],
            [
                "gold_answer_id",
                "gt_answer_id",
                "correct_answer_id",
                "answer_id",
                "predicted_answer_id",
            ],
        )
        if view_value is None or answer_value is None:
            raise ValueError(
                f"{qid}: validation labels must provide golden_view and "
                "gold_answer_id (or a supported alias)"
            )

        gold_view = normalize_view(view_value, f"Validation {qid}")
        gold_answer_id = normalize_answer(answer_value, f"Validation {qid}")
        member["annotation"] = {
            "golden_view": gold_view,
            "gold_answer_id": gold_answer_id,
            "gold_answer_text": member["options"][gold_answer_id],
        }
        members.append(member)

    if missing_labels:
        raise RuntimeError(
            f"Missing validation labels for {len(missing_labels)} questions:\n"
            + "\n".join(missing_labels)
        )

    input_qids = {member["question_id"] for member in members}
    extra_labels = sorted(set(label_by_qid) - input_qids)
    if extra_labels:
        print(
            f"Warning: ignoring {len(extra_labels)} validation labels without "
            "matching input records."
        )
    return members


def load_test_members(stage3_by_qid: Dict[str, dict]) -> List[dict]:
    members: List[dict] = []
    for input_record in read_jsonl(TEST_INPUT_FILE):
        member = normalize_input_record(input_record, "test")
        qid = member["question_id"]
        if qid not in stage3_by_qid:
            raise RuntimeError(f"Stage-3 file is missing test question {qid}")
        member["stage3"] = stage3_by_qid[qid]
        members.append(member)

    test_qids = {member["question_id"] for member in members}
    extra_predictions = sorted(set(stage3_by_qid) - test_qids)
    if extra_predictions:
        print(
            f"Warning: ignoring {len(extra_predictions)} Stage-3 predictions "
            "without matching test inputs."
        )
    return members


# ============================================================
# 6. Candidate groups from base ID and synchronized observations
# ============================================================

def base_scene_id(question_id: str) -> Optional[str]:
    match = BASE_SCENE_RE.match(question_id)
    return match.group(1).lower() if match else None


def scene_signature(member: dict) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (camera, normalize_observation_path(member["views"][camera]))
        for camera in CAMERA_ORDER
    )


def build_candidate_scene_groups(members: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[dict]] = defaultdict(list)
    skipped_non_sfall: List[str] = []

    for member in members:
        base_id = base_scene_id(member["question_id"])
        if base_id is None:
            skipped_non_sfall.append(member["question_id"])
            continue
        grouped[(base_id, scene_signature(member))].append(member)

    candidates: List[dict] = []
    for (base_id, signature), group_members in grouped.items():
        ordered = sorted(group_members, key=lambda item: item["question_id"])
        if len(ordered) < MIN_GROUP_SIZE:
            continue
        if not any(member["split"] == "test" for member in ordered):
            continue

        signature_hash = stable_hash(signature)[:12]
        candidates.append(
            {
                "candidate_group_id": f"{base_id}__obs_{signature_hash}",
                "base_scene_id": base_id,
                "scene_signature_hash": signature_hash,
                "scene_signature": signature,
                "members": ordered,
            }
        )

    candidates.sort(key=lambda item: item["candidate_group_id"])

    if skipped_non_sfall:
        print(
            f"Skipping {len(skipped_non_sfall)} records whose IDs do not match "
            "the sfall_XXXX convention."
        )
    return candidates


# ============================================================
# 7. Text-only semantic filtering and textual priors
# ============================================================

SEMANTIC_FILTER_SYSTEM = """
You are the text-only semantic grouping stage for GoldenViewVQA.

You receive only opaque labels and question stems. All questions already share
the same base scene identifier and exactly the same synchronized six-camera
observation, but that alone is not sufficient reason to transfer information.

Partition every question exactly once. Keep questions together when either:
1. they concern the same core entity and local spatial relation; or
2. they have a sufficiently strong semantic, causal, or behavioral connection
   within the scene.

Separate questions that merely use generic driving vocabulary or happen to
belong to the same scene. Use a singleton group whenever a question lacks a
strong connection to the others. Do not infer answers, camera views, hidden
labels, or unobserved visual facts.

For every partition, produce a concise textual prior derived only from the
stems. It must describe hypotheses and transfer cautions, not ground truth.
""".strip()


def semantic_label_map(candidate: dict) -> Dict[str, dict]:
    return {
        f"Q{index:02d}": member
        for index, member in enumerate(candidate["members"], start=1)
    }


def build_semantic_filter_messages(candidate: dict) -> Tuple[List[dict], Dict[str, dict]]:
    label_to_member = semantic_label_map(candidate)
    stem_lines = [
        f"{label}: {member['question']}"
        for label, member in label_to_member.items()
    ]
    prompt = (
        "Partition the following question stems. Return all opaque labels "
        "exactly once, including singleton partitions when needed.\n\n"
        + "\n\n".join(stem_lines)
    )
    return (
        [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SEMANTIC_FILTER_SYSTEM}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        label_to_member,
    )


def sanitize_string_list(value: Any, limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")
    return [str(item).strip()[:500] for item in value[:limit] if str(item).strip()]


def sanitize_textual_prior(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("textual_prior must be an object")
    return {
        "shared_focus": str(value.get("shared_focus", "")).strip()[:800],
        "core_entities": sanitize_string_list(value.get("core_entities")),
        "local_spatial_relations": sanitize_string_list(
            value.get("local_spatial_relations")
        ),
        "semantic_causal_behavioral_links": sanitize_string_list(
            value.get("semantic_causal_behavioral_links")
        ),
        "transfer_cautions": sanitize_string_list(value.get("transfer_cautions")),
    }


def validate_semantic_filter_output(data: dict, valid_labels: Sequence[str]) -> dict:
    valid_set = set(valid_labels)
    label_order = {label: index for index, label in enumerate(valid_labels)}
    raw_groups = data.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("Semantic filter returned no groups")

    seen: set[str] = set()
    groups: List[dict] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ValueError("Each semantic group must be an object")
        raw_labels = raw_group.get("member_labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise ValueError("Each semantic group must contain member_labels")

        labels = [str(label).strip() for label in raw_labels]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate label within semantic group: {labels}")
        unknown = set(labels) - valid_set
        if unknown:
            raise ValueError(f"Unknown semantic labels: {sorted(unknown)}")
        overlap = seen.intersection(labels)
        if overlap:
            raise ValueError(f"Labels assigned to multiple groups: {sorted(overlap)}")
        seen.update(labels)

        labels.sort(key=label_order.__getitem__)
        groups.append(
            {
                "member_labels": labels,
                "textual_prior": sanitize_textual_prior(
                    raw_group.get("textual_prior")
                ),
            }
        )

    missing = valid_set - seen
    if missing:
        raise ValueError(f"Semantic filter omitted labels: {sorted(missing)}")

    groups.sort(key=lambda group: label_order[group["member_labels"][0]])
    return {"groups": groups}


def semantic_filter_fingerprint(candidate: dict) -> str:
    return stable_hash(
        {
            "version": SEMANTIC_FILTER_VERSION,
            "model": MODEL_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "minimum_group_size": MIN_GROUP_SIZE,
            "scene_signature": candidate["scene_signature"],
            "members": [
                {
                    "question_id": member["question_id"],
                    "question": member["question"],
                }
                for member in candidate["members"]
            ],
        }
    )


def run_semantic_filter(client: OpenAI, candidate: dict) -> dict:
    messages, label_to_member = build_semantic_filter_messages(candidate)
    labels = list(label_to_member)
    response = call_structured(
        client=client,
        input_messages=messages,
        output_schema=SemanticFilterOutput,
        operation_name=f"Semantic filter {candidate['candidate_group_id']}",
        validator=lambda data: validate_semantic_filter_output(data, labels),
    )

    partitions: List[dict] = []
    discarded_singletons: List[str] = []
    for group in response["parsed"]["groups"]:
        question_ids = [
            label_to_member[label]["question_id"]
            for label in group["member_labels"]
        ]
        semantic_hash = stable_hash(sorted(question_ids))[:10]
        partition = {
            "semantic_group_id": (
                f"{candidate['candidate_group_id']}__sem_{semantic_hash}"
            ),
            "member_question_ids": question_ids,
            "group_size": len(question_ids),
            "split_counts": dict(
                Counter(
                    label_to_member[label]["split"]
                    for label in group["member_labels"]
                )
            ),
            "retained": len(question_ids) >= MIN_GROUP_SIZE,
            "textual_prior": group["textual_prior"],
        }
        partitions.append(partition)
        if not partition["retained"]:
            discarded_singletons.extend(question_ids)

    return {
        "candidate_group_id": candidate["candidate_group_id"],
        "base_scene_id": candidate["base_scene_id"],
        "scene_signature_hash": candidate["scene_signature_hash"],
        "candidate_member_question_ids": [
            member["question_id"] for member in candidate["members"]
        ],
        "candidate_group_size": len(candidate["members"]),
        "filter_version": SEMANTIC_FILTER_VERSION,
        "filter_fingerprint": semantic_filter_fingerprint(candidate),
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "semantic_partitions": partitions,
        "discarded_singleton_question_ids": discarded_singletons,
        "raw_output": response["raw_output"],
        "response_id": response["response_id"],
        "usage": response["usage"],
        "attempt": response["attempt"],
    }


def cached_semantic_filter_is_valid(record: dict, candidate: dict) -> bool:
    try:
        expected = {
            member["question_id"] for member in candidate["members"]
        }
        partitions = record["semantic_partitions"]
        if not isinstance(partitions, list) or not partitions:
            return False

        seen: set[str] = set()
        for partition in partitions:
            question_ids = partition["member_question_ids"]
            if not isinstance(question_ids, list) or not question_ids:
                return False
            if len(question_ids) != len(set(question_ids)):
                return False
            if seen.intersection(question_ids):
                return False
            if not set(question_ids).issubset(expected):
                return False
            seen.update(question_ids)

            if partition.get("group_size") != len(question_ids):
                return False
            if bool(partition.get("retained")) != (
                len(question_ids) >= MIN_GROUP_SIZE
            ):
                return False
            if not isinstance(partition.get("semantic_group_id"), str):
                return False
            sanitize_textual_prior(partition.get("textual_prior"))

        return seen == expected
    except (KeyError, TypeError, ValueError):
        return False


def collect_retained_groups(
    candidates: Sequence[dict],
    filter_records: Dict[str, dict],
) -> List[dict]:
    retained: List[dict] = []
    seen_qids: set[str] = set()

    for candidate in candidates:
        candidate_id = candidate["candidate_group_id"]
        filter_record = filter_records[candidate_id]
        member_by_qid = {
            member["question_id"]: member for member in candidate["members"]
        }
        for partition in filter_record["semantic_partitions"]:
            if not partition.get("retained"):
                continue

            question_ids = partition["member_question_ids"]
            unknown = set(question_ids) - set(member_by_qid)
            if unknown:
                raise RuntimeError(
                    f"Cached semantic group contains unknown IDs: {sorted(unknown)}"
                )
            overlap = seen_qids.intersection(question_ids)
            if overlap:
                raise RuntimeError(
                    f"Questions occur in multiple retained groups: {sorted(overlap)}"
                )
            seen_qids.update(question_ids)

            members = [member_by_qid[qid] for qid in question_ids]
            retained.append(
                {
                    "semantic_group_id": partition["semantic_group_id"],
                    "candidate_group_id": candidate_id,
                    "base_scene_id": candidate["base_scene_id"],
                    "scene_signature_hash": candidate["scene_signature_hash"],
                    "members": members,
                    "textual_prior": partition["textual_prior"],
                }
            )

    retained.sort(key=lambda group: group["semantic_group_id"])
    return retained


# ============================================================
# 8. Shared images and joint group-level verification
# ============================================================

GROUP_VERIFY_SYSTEM = """
You are the Stage-4 cross-split group-level verifier for GoldenViewVQA.

The subgroup was retained by a text-only semantic filter. You receive its
shared six synchronized camera views, all member questions and answer options,
the current Stage-3 predictions for test members, a textual group prior, and
ground-truth annotations for validation members when available.

Jointly re-evaluate every test member. The textual prior is a soft hypothesis.
Validation annotations are trusted only for their validation questions. When a
validation and test question concern the same core entity or local spatial
configuration, use the validation answer and supporting view as a strong
reference and favor supporting-view consistency. This is not a hard constraint:
select a different view when clear question-specific visual evidence makes it
more decisive. Never copy an answer ID across questions because option IDs are
local to each question.

Choose the single camera containing the clearest decisive evidence for each
answer. Use NONE_OF_THE_ABOVE only when none of the six images provides enough
direct evidence. Treat the Stage-3 pair as a strong current prediction and
change it only when the combined visual and group evidence is stronger.

Return exactly one prediction for every test label and no prediction for a
validation label. Keep answer and supporting view mutually consistent.
""".strip()


def resolve_image_path(relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = NUSCENES_ROOT / path
    return path.resolve()


def encode_image_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_types.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")

    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_group_context(group: dict) -> dict:
    validation_members = sorted(
        [member for member in group["members"] if member["split"] == "validation"],
        key=lambda member: member["question_id"],
    )
    test_members = sorted(
        [member for member in group["members"] if member["split"] == "test"],
        key=lambda member: member["question_id"],
    )

    label_to_member: Dict[str, dict] = {}
    member_payload: List[dict] = []
    validation_prior: List[dict] = []

    for index, member in enumerate(validation_members, start=1):
        label = f"V{index:02d}"
        label_to_member[label] = member
        member_payload.append(
            {
                "label": label,
                "split": "validation",
                "reasoning_type": member["question_group"],
                "question": member["question"],
                "options": member["options"],
            }
        )
        annotation = member["annotation"]
        validation_prior.append(
            {
                "question_label": label,
                "gold_answer_id": annotation["gold_answer_id"],
                "gold_answer_text": annotation["gold_answer_text"],
                "golden_view": annotation["golden_view"],
            }
        )

    test_labels: List[str] = []
    for index, member in enumerate(test_members, start=1):
        label = f"T{index:02d}"
        test_labels.append(label)
        label_to_member[label] = member
        member_payload.append(
            {
                "label": label,
                "split": "test",
                "reasoning_type": member["question_group"],
                "question": member["question"],
                "options": member["options"],
                "current_stage3_prediction": {
                    "predicted_view": member["stage3"]["predicted_view"],
                    "predicted_answer_id": member["stage3"]["predicted_answer_id"],
                    "predicted_answer_text": member["options"][
                        member["stage3"]["predicted_answer_id"]
                    ],
                },
            }
        )

    return {
        "label_to_member": label_to_member,
        "test_labels": test_labels,
        "member_payload": member_payload,
        "group_prior": {
            "textual_group_prior": group["textual_prior"],
            "validation_derived_prior": validation_prior,
        },
    }


def build_group_verify_messages(group: dict, context: dict) -> List[dict]:
    prompt = f"""
Semantic group: {group['semantic_group_id']}

COMPLETE GROUP-LEVEL PRIOR:
{json.dumps(context['group_prior'], ensure_ascii=False, indent=2)}

GROUP MEMBERS:
{json.dumps(context['member_payload'], ensure_ascii=False, indent=2)}

Inspect the shared six-view scene below and jointly return the final answer-view
pair for every test label: {', '.join(context['test_labels'])}.

For validation_prior_relation, use:
- consistent: a relevant validation annotation supports the final reasoning;
- question_specific_override: a relevant validation reference exists, but clear
  question-specific evidence justifies different decisive evidence;
- not_applicable: no validation prior exists or none is directly relevant.
""".strip()

    content: List[dict] = [{"type": "input_text", "text": prompt}]
    reference_member = group["members"][0]
    for camera in CAMERA_ORDER:
        image_path = resolve_image_path(reference_member["views"][camera])
        content.append(
            {
                "type": "input_text",
                "text": f"The following synchronized image is {camera}.",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_data_url(image_path),
                "detail": IMAGE_DETAIL,
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": "Return the structured group verification now.",
        }
    )
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": GROUP_VERIFY_SYSTEM}],
        },
        {"role": "user", "content": content},
    ]


def validate_group_verification_output(
    data: dict,
    test_labels: Sequence[str],
    has_validation_prior: bool,
) -> dict:
    expected = set(test_labels)
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Group verifier predictions must be a list")

    seen: set[str] = set()
    cleaned: List[dict] = []
    order = {label: index for index, label in enumerate(test_labels)}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Each group prediction must be an object")
        label = str(prediction.get("question_label", "")).strip()
        if label not in expected:
            raise ValueError(f"Unexpected test label in group output: {label!r}")
        if label in seen:
            raise ValueError(f"Duplicate group prediction for {label}")
        seen.add(label)

        relation = str(prediction.get("validation_prior_relation", "")).strip()
        if relation not in {
            "consistent",
            "question_specific_override",
            "not_applicable",
        }:
            raise ValueError(f"Invalid validation_prior_relation: {relation!r}")
        if not has_validation_prior and relation != "not_applicable":
            raise ValueError(
                "Test-only groups must use validation_prior_relation=not_applicable"
            )

        cleaned.append(
            {
                "question_label": label,
                "predicted_view": normalize_view(
                    prediction.get("predicted_view"),
                    f"Group prediction {label}",
                ),
                "predicted_answer_id": normalize_answer(
                    prediction.get("predicted_answer_id"),
                    f"Group prediction {label}",
                ),
                "validation_prior_relation": relation,
                "evidence_summary": str(
                    prediction.get("evidence_summary", "")
                ).strip()[:1600],
            }
        )

    missing = expected - seen
    if missing:
        raise ValueError(f"Group verifier omitted test labels: {sorted(missing)}")

    cleaned.sort(key=lambda item: order[item["question_label"]])
    return {
        "group_consistency_summary": str(
            data.get("group_consistency_summary", "")
        ).strip()[:2000],
        "predictions": cleaned,
    }


def group_audit_fingerprint(group: dict, context: dict) -> str:
    reference_member = group["members"][0]
    return stable_hash(
        {
            "version": GROUP_VERIFY_VERSION,
            "model": MODEL_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "image_detail": IMAGE_DETAIL,
            "semantic_group_id": group["semantic_group_id"],
            "scene_signature": scene_signature(reference_member),
            "group_prior": context["group_prior"],
            "member_payload": context["member_payload"],
        }
    )


def run_group_verification(client: OpenAI, group: dict) -> dict:
    context = build_group_context(group)
    if not context["test_labels"]:
        raise ValueError(
            f"Semantic group {group['semantic_group_id']} has no test member"
        )

    has_validation_prior = bool(
        context["group_prior"]["validation_derived_prior"]
    )
    messages = build_group_verify_messages(group, context)
    response = call_structured(
        client=client,
        input_messages=messages,
        output_schema=GroupVerificationOutput,
        operation_name=f"Group verification {group['semantic_group_id']}",
        validator=lambda data: validate_group_verification_output(
            data,
            context["test_labels"],
            has_validation_prior,
        ),
    )

    predictions: List[dict] = []
    for prediction in response["parsed"]["predictions"]:
        member = context["label_to_member"][prediction["question_label"]]
        stage3 = member["stage3"]
        final_pair = {
            "question_id": member["question_id"],
            "predicted_view": prediction["predicted_view"],
            "predicted_answer_id": prediction["predicted_answer_id"],
        }
        predictions.append(
            {
                **final_pair,
                "stage3": stage3,
                "changed": (
                    final_pair["predicted_view"] != stage3["predicted_view"]
                    or final_pair["predicted_answer_id"]
                    != stage3["predicted_answer_id"]
                ),
                "validation_prior_relation": prediction[
                    "validation_prior_relation"
                ],
                "evidence_summary": prediction["evidence_summary"],
            }
        )

    validation_annotations = []
    for label, member in context["label_to_member"].items():
        if member["split"] == "validation":
            validation_annotations.append(
                {
                    "question_label": label,
                    "question_id": member["question_id"],
                    **member["annotation"],
                }
            )

    return {
        "semantic_group_id": group["semantic_group_id"],
        "candidate_group_id": group["candidate_group_id"],
        "base_scene_id": group["base_scene_id"],
        "scene_signature_hash": group["scene_signature_hash"],
        "member_question_ids": [
            member["question_id"] for member in group["members"]
        ],
        "test_question_ids": [
            prediction["question_id"] for prediction in predictions
        ],
        "group_type": (
            "cross_split" if validation_annotations else "test_only"
        ),
        "verify_version": GROUP_VERIFY_VERSION,
        "audit_fingerprint": group_audit_fingerprint(group, context),
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "image_detail": IMAGE_DETAIL,
        "textual_group_prior": group["textual_prior"],
        "validation_derived_prior": validation_annotations,
        "group_consistency_summary": response["parsed"][
            "group_consistency_summary"
        ],
        "stage4_predictions": predictions,
        "raw_output": response["raw_output"],
        "response_id": response["response_id"],
        "usage": response["usage"],
        "attempt": response["attempt"],
    }


def cached_group_audit_is_valid(record: dict, group: dict) -> bool:
    try:
        expected_members = {
            member["question_id"]: member
            for member in group["members"]
            if member["split"] == "test"
        }
        predictions = record["stage4_predictions"]
        if not isinstance(predictions, list):
            return False

        seen: set[str] = set()
        for prediction in predictions:
            qid = prediction["question_id"]
            if qid not in expected_members or qid in seen:
                return False
            seen.add(qid)
            if prediction.get("predicted_view") not in VALID_VIEWS:
                return False
            if prediction.get("predicted_answer_id") not in VALID_ANSWERS:
                return False

            stage3 = expected_members[qid]["stage3"]
            expected_changed = (
                prediction["predicted_view"] != stage3["predicted_view"]
                or prediction["predicted_answer_id"]
                != stage3["predicted_answer_id"]
            )
            if bool(prediction.get("changed")) != expected_changed:
                return False

        return seen == set(expected_members)
    except (KeyError, TypeError):
        return False


# ============================================================
# 9. Cache orchestration and final predictions
# ============================================================

def prepare_semantic_filters(
    candidates: Sequence[dict],
    get_client: Callable[[], OpenAI],
) -> Dict[str, dict]:
    if REGENERATE_SEMANTIC_GROUPS and STAGE4_SEMANTIC_GROUP_FILE.exists():
        STAGE4_SEMANTIC_GROUP_FILE.unlink()

    existing = load_latest_jsonl_map(
        STAGE4_SEMANTIC_GROUP_FILE,
        "candidate_group_id",
    )
    current: Dict[str, dict] = {}

    for candidate in tqdm(candidates, desc="Stage-4 semantic filtering"):
        candidate_id = candidate["candidate_group_id"]
        fingerprint = semantic_filter_fingerprint(candidate)
        cached = existing.get(candidate_id)
        if (
            cached is not None
            and cached.get("filter_fingerprint") == fingerprint
            and cached.get("filter_version") == SEMANTIC_FILTER_VERSION
            and cached_semantic_filter_is_valid(cached, candidate)
        ):
            current[candidate_id] = cached
            continue

        record = run_semantic_filter(get_client(), candidate)
        append_jsonl(STAGE4_SEMANTIC_GROUP_FILE, record)
        current[candidate_id] = record

    return current


def prepare_group_audits(
    groups: Sequence[dict],
    get_client: Callable[[], OpenAI],
) -> Dict[str, dict]:
    if (
        (REGENERATE_GROUP_AUDITS or not RESUME)
        and STAGE4_GROUP_AUDIT_FILE.exists()
    ):
        STAGE4_GROUP_AUDIT_FILE.unlink()

    existing = load_latest_jsonl_map(
        STAGE4_GROUP_AUDIT_FILE,
        "semantic_group_id",
    )
    current: Dict[str, dict] = {}

    for group in tqdm(groups, desc="Stage-4 joint group verification"):
        group_id = group["semantic_group_id"]
        context = build_group_context(group)
        fingerprint = group_audit_fingerprint(group, context)
        cached = existing.get(group_id)
        if (
            RESUME
            and not REGENERATE_GROUP_AUDITS
            and cached is not None
            and cached.get("audit_fingerprint") == fingerprint
            and cached.get("verify_version") == GROUP_VERIFY_VERSION
            and cached_group_audit_is_valid(cached, group)
        ):
            current[group_id] = cached
            continue

        audit = run_group_verification(get_client(), group)
        append_jsonl(STAGE4_GROUP_AUDIT_FILE, audit)
        current[group_id] = audit

        changed_count = sum(
            prediction["changed"] for prediction in audit["stage4_predictions"]
        )
        print(
            f"\n{group_id}: {audit['group_type']}, "
            f"{changed_count}/{len(audit['stage4_predictions'])} changed"
        )

    return current


def apply_group_audits(
    test_members: Sequence[dict],
    eligible_groups: Sequence[dict],
    audits: Dict[str, dict],
) -> Tuple[List[dict], List[dict]]:
    replacement_by_qid: Dict[str, dict] = {}
    selected: List[dict] = []

    for group in eligible_groups:
        group_id = group["semantic_group_id"]
        audit = audits.get(group_id)
        if audit is None:
            raise RuntimeError(f"Missing group audit for {group_id}")

        for prediction in audit["stage4_predictions"]:
            qid = prediction["question_id"]
            if qid in replacement_by_qid:
                raise RuntimeError(f"Multiple Stage-4 predictions for {qid}")
            final_prediction = {
                "question_id": qid,
                "predicted_view": prediction["predicted_view"],
                "predicted_answer_id": prediction["predicted_answer_id"],
            }
            replacement_by_qid[qid] = final_prediction

            if prediction["changed"]:
                selected.append(
                    {
                        "question_id": qid,
                        "semantic_group_id": group_id,
                        "group_type": audit["group_type"],
                        "stage3": prediction["stage3"],
                        "stage4": final_prediction,
                        "validation_prior_relation": prediction[
                            "validation_prior_relation"
                        ],
                        "evidence_summary": prediction["evidence_summary"],
                        "group_consistency_summary": audit[
                            "group_consistency_summary"
                        ],
                        "related_validation_annotations": audit[
                            "validation_derived_prior"
                        ],
                    }
                )

    final_records: List[dict] = []
    for member in test_members:
        qid = member["question_id"]
        final_records.append(
            replacement_by_qid.get(
                qid,
                {
                    "question_id": qid,
                    "predicted_view": member["stage3"]["predicted_view"],
                    "predicted_answer_id": member["stage3"][
                        "predicted_answer_id"
                    ],
                },
            )
        )

    selected.sort(key=lambda item: item["question_id"])
    return final_records, selected


# ============================================================
# 10. Main
# ============================================================

def validate_runtime_configuration() -> None:
    if MIN_GROUP_SIZE < 2:
        raise ValueError("MIN_GROUP_SIZE must be at least 2")
    if MAX_API_RETRIES < 1:
        raise ValueError("MAX_API_RETRIES must be at least 1")
    if MAX_OUTPUT_TOKENS < 1 or MAX_RETRY_OUTPUT_TOKENS < 1:
        raise ValueError("Output-token limits must be positive")
    if API_TIMEOUT <= 0 or RETRY_BASE_SLEEP < 0:
        raise ValueError("API_TIMEOUT must be positive and retry sleep nonnegative")
    if IMAGE_DETAIL not in {"auto", "low", "high"}:
        raise ValueError("IMAGE_DETAIL must be auto, low, or high")
    if REASONING_EFFORT not in {
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError(f"Unsupported REASONING_EFFORT: {REASONING_EFFORT}")
    if MAX_GROUPS is not None and MAX_GROUPS <= 0:
        raise ValueError("MAX_GROUPS must be positive")


def main() -> None:
    validate_runtime_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage3_by_qid = load_prediction_map(STAGE3_FILE)
    validation_members = load_validation_members()
    test_members = load_test_members(stage3_by_qid)

    val_qids = {member["question_id"] for member in validation_members}
    test_qids = {member["question_id"] for member in test_members}
    overlap = val_qids.intersection(test_qids)
    if overlap:
        raise RuntimeError(
            f"Validation/test question IDs overlap: {sorted(overlap)}"
        )

    combined_members = validation_members + test_members
    all_candidates = build_candidate_scene_groups(combined_members)
    candidates = (
        all_candidates[:MAX_GROUPS] if MAX_GROUPS is not None else all_candidates
    )

    candidate_split_types = Counter()
    for candidate in all_candidates:
        splits = {member["split"] for member in candidate["members"]}
        candidate_split_types[
            "cross_split" if len(splits) > 1 else "test_only"
        ] += 1

    print("\n" + "=" * 84)
    print("GoldenViewVQA Stage 4: cross-split GPT-5.6 group verification")
    print("=" * 84)
    print("Validation inputs:", VAL_INPUT_FILE)
    print("Validation labels:", VAL_LABEL_FILE)
    print("Test inputs:", TEST_INPUT_FILE)
    print("Stage-3 predictions:", STAGE3_FILE)
    print("Validation questions:", len(validation_members))
    print("Test questions:", len(test_members))
    print("Candidate scene groups:", len(all_candidates))
    print("Candidate split types:", dict(candidate_split_types))
    print("Model:", MODEL_NAME)
    print("Minimum retained group size:", MIN_GROUP_SIZE)
    if MAX_GROUPS is not None:
        print("Partial-run candidate limit:", MAX_GROUPS)
    print("=" * 84 + "\n")

    client_holder: Dict[str, OpenAI] = {}

    def get_client() -> OpenAI:
        if "client" not in client_holder:
            client_holder["client"] = build_client()
        return client_holder["client"]

    if candidates:
        filter_records = prepare_semantic_filters(candidates, get_client)
        retained_groups = collect_retained_groups(candidates, filter_records)
    else:
        retained_groups = []

    eligible_groups = [
        group
        for group in retained_groups
        if any(member["split"] == "test" for member in group["members"])
    ]
    retained_type_counts = Counter(
        "cross_split"
        if any(member["split"] == "validation" for member in group["members"])
        else "test_only"
        for group in eligible_groups
    )

    print("Retained semantic groups:", len(retained_groups))
    print("Test-bearing semantic groups:", len(eligible_groups))
    print("Retained split types:", dict(retained_type_counts))

    audits = prepare_group_audits(eligible_groups, get_client)

    if MAX_GROUPS is not None and len(candidates) < len(all_candidates):
        print(
            "\nPartial Stage-4 run finished. Semantic and audit caches were "
            "saved, but no final submission was written. Unset MAX_GROUPS "
            "and MAX_SAMPLES for the full run."
        )
        return

    final_records, selected = apply_group_audits(
        test_members=test_members,
        eligible_groups=eligible_groups,
        audits=audits,
    )
    write_jsonl(STAGE4_SELECTED_FILE, selected)
    write_jsonl(STAGE4_OUTPUT_FILE, final_records)

    change_types = Counter()
    for record in selected:
        old = record["stage3"]
        new = record["stage4"]
        view_changed = old["predicted_view"] != new["predicted_view"]
        answer_changed = old["predicted_answer_id"] != new["predicted_answer_id"]
        if view_changed and answer_changed:
            change_types["answer_and_view"] += 1
        elif view_changed:
            change_types["view_only"] += 1
        elif answer_changed:
            change_types["answer_only"] += 1

    print("\n" + "=" * 84)
    print("Stage 4 finished")
    print("=" * 84)
    print("Applied group-level changes:", len(selected))
    print("Change types:", dict(change_types))
    print("Semantic group cache:", STAGE4_SEMANTIC_GROUP_FILE)
    print("Group audit cache:", STAGE4_GROUP_AUDIT_FILE)
    print("Selected changes:", STAGE4_SELECTED_FILE)
    print("Final predictions:", STAGE4_OUTPUT_FILE)

    for index, record in enumerate(selected, start=1):
        old = record["stage3"]
        new = record["stage4"]
        print(
            f"{index:02d}. {record['question_id']}: "
            f"{old['predicted_view']}/{old['predicted_answer_id']} -> "
            f"{new['predicted_view']}/{new['predicted_answer_id']} "
            f"[{record['group_type']}]"
        )


if __name__ == "__main__":
    main()
