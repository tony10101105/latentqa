"""
Dataset loaders for debiasing evaluations.
"""

from typing import Dict, List
import csv
import io
import urllib.request

from datasets import load_dataset


_CROWS_PAIRS_CSV_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/"
    "crows_pairs_anonymized.csv"
)


def _normalize_crows_pairs_row(row: Dict) -> Dict:
    sent_more = row.get("sent_more") or row.get("sentence_more") or row.get("stereotype")
    sent_less = row.get("sent_less") or row.get("sentence_less") or row.get("anti_stereotype")
    if sent_more is None or sent_less is None:
        raise ValueError("CrowS-Pairs row missing sent_more/sent_less fields.")

    return {
        "id": row.get("id"),
        "bias_type": row.get("bias_type") or row.get("category"),
        "sent_more": sent_more,
        "sent_less": sent_less,
    }


def load_crows_pairs(split: str = "test") -> List[Dict]:
    try:
        dataset = load_dataset("crows_pairs", split=split)
        return [_normalize_crows_pairs_row(row) for row in dataset]
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" not in message:
            raise
        return _load_crows_pairs_from_csv()


def _load_crows_pairs_from_csv() -> List[Dict]:
    with urllib.request.urlopen(_CROWS_PAIRS_CSV_URL) as response:
        content = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        rows.append(
            {
                "id": row.get("id"),
                "bias_type": row.get("bias_type"),
                "sent_more": row.get("sent_more"),
                "sent_less": row.get("sent_less"),
            }
        )
    return [_normalize_crows_pairs_row(row) for row in rows]


def load_debias_dataset(name: str, split: str = "test") -> List[Dict]:
    name = name.lower()
    if name in {"crows_pairs", "crowspairs", "crows-pairs"}:
        return load_crows_pairs(split=split)
    raise ValueError(f"Unsupported debias dataset: {name}")
