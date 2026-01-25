"""
Build decoder-steering control files for spurious-feature interventions.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional

import fire


def _slugify(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    return value or "spurious_feature"


def build_spurious_feature_control(
    spurious_feature: str,
    output_file: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> str:
    feature = spurious_feature.strip().replace("_", " ")
    qa_pairs: List[Tuple[str, str]] = [
        (f"Is {feature} relevant to answering this question?", "No."),
        (f"Does the answer depend on {feature}?", "No."),
        (f"Should {feature} influence the answer?", "No."),
    ]
    if max_pairs is not None:
        qa_pairs = qa_pairs[:max_pairs]

    payload: Dict[str, List[List[str]]] = {
        f"Spurious feature steering: {feature}": [[q, a] for q, a in qa_pairs]
    }

    if output_file is None:
        output_file = os.path.join("controls", f"spurious_feature_{_slugify(feature)}.json")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_file


def main(
    spurious_feature: str = "hair color",
    output_file: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> str:
    return build_spurious_feature_control(
        spurious_feature=spurious_feature,
        output_file=output_file,
        max_pairs=max_pairs,
    )


if __name__ == "__main__":
    fire.Fire(main)
