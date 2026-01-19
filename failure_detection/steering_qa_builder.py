"""
Build decoder-steering QA pairs from a failure case.

Given a failure case (optionally selected by sample_id), this script produces a
controls/*.json file compatible with lit/control.py, containing QA pairs that
express the desired behavior (e.g., "hair color is irrelevant", or correct answer).
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import fire


SPURIOUS_FALLBACK = "the mentioned demographic attribute"
SPURIOUS_KEYWORDS = [
    "hair color",
    "hair_colour",
    "age",
    "gender",
    "sex",
    "ethnicity",
    "race",
    "nationality",
]


def _load_failures(failures_file: str) -> List[Dict[str, Any]]:
    with open(failures_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_failure(
    failures: List[Dict[str, Any]], sample_id: Optional[int], seed: int
) -> Dict[str, Any]:
    if sample_id is not None:
        for failure in failures:
            if failure.get("sample_id") == sample_id:
                return failure
        raise ValueError(f"sample_id {sample_id} not found in failures.")
    rng = random.Random(seed)
    return rng.choice(failures)


def _normalize_ground_truth(failure: Dict[str, Any]) -> Optional[str]:
    ground_truth = failure.get("ground_truth", None)
    choices = failure.get("choices") or []
    if ground_truth is None:
        return None

    if isinstance(ground_truth, int):
        if 0 <= ground_truth < len(choices):
            return choices[ground_truth]
        return None

    if isinstance(ground_truth, str):
        stripped = ground_truth.strip()
        if stripped.isdigit():
            idx = int(stripped)
            if 0 <= idx < len(choices):
                return choices[idx]
        if stripped in choices:
            return stripped
    return None


def _extract_spurious_feature(failure: Dict[str, Any]) -> str:
    issues = failure.get("issues", [])
    for issue in issues:
        if issue.get("type") == "spurious_feature":
            name = issue.get("spurious_feature_name")
            if name:
                return name.replace("_", " ")

    haystack = " ".join(
        [
            failure.get("question", ""),
            failure.get("model_response", ""),
            failure.get("judgment_summary", ""),
        ]
    ).lower()
    for keyword in SPURIOUS_KEYWORDS:
        if keyword in haystack:
            return keyword.replace("_", " ")
    return SPURIOUS_FALLBACK


def _issue_types(failure: Dict[str, Any]) -> List[str]:
    return [issue.get("type", "") for issue in failure.get("issues", [])]


def build_qa_pairs(
    failure: Dict[str, Any], max_pairs: Optional[int] = None
) -> List[Tuple[str, str]]:
    qa_pairs: List[Tuple[str, str]] = []
    issue_types = _issue_types(failure)

    if "spurious_feature" in issue_types:
        feature = _extract_spurious_feature(failure)
        qa_pairs.append((f"Is {feature} relevant to answering this question?", "No."))
        qa_pairs.append((f"Does the answer depend on {feature}?", "No."))

    if "incorrect_answer" in issue_types or "reasoning_failure" in issue_types:
        correct_text = _normalize_ground_truth(failure)
        if correct_text:
            qa_pairs.append((f"Is the correct answer: {correct_text}?", "Yes."))
            qa_pairs.append(("What is the correct answer?", correct_text))

    if not qa_pairs:
        qa_pairs.append(("Is the model's answer correct?", "No."))

    if max_pairs is not None:
        qa_pairs = qa_pairs[:max_pairs]
    return qa_pairs


def build_control_payload(
    failure: Dict[str, Any], qa_pairs: List[Tuple[str, str]]
) -> Dict[str, List[List[str]]]:
    sample_id = failure.get("sample_id", "unknown")
    question = failure.get("question", "unknown question")
    control_key = f"Failure case {sample_id}: {question}"
    return {control_key: [[q, a] for q, a in qa_pairs]}


def main(
    failures_file: str = "failure_detection_output/failures.json",
    output_file: Optional[str] = None,
    sample_id: Optional[int] = None,
    max_pairs: Optional[int] = None,
    seed: int = 7,
) -> str:
    failures = _load_failures(failures_file)
    if not failures:
        raise ValueError("No failures found in failures file.")
    failure = _pick_failure(failures, sample_id=sample_id, seed=seed)
    qa_pairs = build_qa_pairs(failure, max_pairs=max_pairs)
    payload = build_control_payload(failure, qa_pairs)

    if output_file is None:
        sample_id_value = failure.get("sample_id", "unknown")
        output_file = os.path.join("controls", f"failure_{sample_id_value}.json")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_file


if __name__ == "__main__":
    fire.Fire(main)
