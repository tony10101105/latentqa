"""
Pipeline to:
1) detect gender-biased responses via LLM judge (baseline),
2) steer the model to reduce reliance on gender as a spurious feature,
3) report only the samples where gender bias was corrected after steering,
4) report samples where gender was relevant but steering caused errors.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from lit.configs.steer_config import steer_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
from lit.control import steer as lit_steer

from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name
from failure_detection.judge_evaluator import JudgeEvaluator
from failure_detection.model_evaluator import ModelEvaluator
from failure_detection.oversteering_judge import OversteeringJudgeEvaluator
from failure_detection.steered_model_evaluator import SteeredModelEvaluator
from failure_detection.steering_control_builder import build_spurious_feature_control

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None


def _steer_model(
    control_name: str,
    steer_dataset: str,
    steer_samples: int,
    target_model_name: str,
    decoder_model_name: str,
    device: str,
) -> str:
    args = steer_config()
    update_config(
        args,
        control=control_name,
        dataset=steer_dataset,
        samples=steer_samples,
        target_model_name=target_model_name,
        decoder_model_name=decoder_model_name,
        save_model=True,
        eval_prompts="",
    )
    tokenizer = get_tokenizer(args.target_model_name)
    decoder_model = get_model(
        model_name=args.target_model_name,
        tokenizer=tokenizer,
        load_peft_checkpoint=args.decoder_model_name,
        device=device,
    )
    lit_steer(args, decoder_model, tokenizer, device=device)
    return os.path.join("out", "model", f"steer_{control_name}_{steer_dataset}_{steer_samples}")


def _get_gender_bias_issue(
    judgment: Dict[str, any],
    spurious_feature: str,
) -> Tuple[bool, Optional[Dict[str, str]]]:
    issues = judgment.get("issues", []) or []
    feature = spurious_feature.lower()
    for issue in issues:
        if issue.get("type") != "spurious_feature":
            continue
        name = (issue.get("spurious_feature_name") or "").lower()
        if not name or feature in name:
            return True, issue
    if judgment.get("has_issues") and not issues:
        return True, None
    return False, None


def _write_excel(
    output_path: str,
    sheets: Dict[str, Tuple[List[str], List[Dict[str, any]]]],
) -> None:
    if Workbook is None:
        raise ImportError(
            "openpyxl is required to write Excel output. Install with: pip install openpyxl"
        )
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    for sheet_name, (columns, rows) in sheets.items():
        ws = wb.create_sheet(title=sheet_name)
        ws.append(columns)
        for row in rows:
            ws.append([row.get(col, "") for col in columns])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wb.save(output_path)


def run_pipeline(
    output_dir: str,
    base_model_name: str,
    decoder_model_name: str,
    steer_dataset: str,
    steer_samples: int,
    benchmark_name: str,
    max_samples: Optional[int],
    spurious_feature: str,
    judge_model: str,
    device: Optional[str],
) -> Dict[str, any]:
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("GENDER BIAS CORRECTION + OVERSTEERING (JUDGE-BASED)")
    print("=" * 80)

    benchmark = get_benchmark_by_name(benchmark_name)
    if not benchmark:
        raise ValueError(f"Benchmark not found: {benchmark_name}")

    print("\n[Step 1] Load benchmark samples...")
    samples = load_benchmark(benchmark, max_samples=max_samples)
    samples_file = os.path.join(output_dir, "benchmark_samples.json")
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print("\n[Step 2] Evaluate baseline model...")
    baseline_evaluator = ModelEvaluator(model_name=base_model_name, device=device)
    baseline_eval_file = os.path.join(output_dir, "baseline_evaluations.json")
    baseline_results = baseline_evaluator.evaluate_samples(
        samples,
        output_file=baseline_eval_file,
    )

    print("\n[Step 3] Judge baseline for gender bias...")
    judge = JudgeEvaluator(judge_model=judge_model, device=device)
    spurious_map = {benchmark.name: [spurious_feature]}
    baseline_judgments_file = os.path.join(output_dir, "baseline_gender_bias_judgments.json")
    baseline_judgments = judge.evaluate_responses(
        baseline_results,
        spurious_features_map=spurious_map,
        output_file=baseline_judgments_file,
        failure_types=["spurious_feature"],
    )

    print("\n[Step 4] Steer model to reduce gender bias...")
    control_file = build_spurious_feature_control(spurious_feature=spurious_feature)
    control_name = os.path.splitext(os.path.basename(control_file))[0]
    steered_model_dir = _steer_model(
        control_name=control_name,
        steer_dataset=steer_dataset,
        steer_samples=steer_samples,
        target_model_name=base_model_name,
        decoder_model_name=decoder_model_name,
        device=device,
    )
    print(f"Steered model saved to {steered_model_dir}")

    print("\n[Step 5] Evaluate steered model...")
    steered_evaluator = SteeredModelEvaluator(
        base_model_name=base_model_name,
        peft_checkpoint=steered_model_dir,
        device=device,
    )
    steered_eval_file = os.path.join(output_dir, "steered_evaluations.json")
    steered_results = steered_evaluator.evaluate_samples(
        samples,
        output_file=steered_eval_file,
    )

    print("\n[Step 6] Judge steered model for gender bias...")
    steered_judgments_file = os.path.join(output_dir, "steered_gender_bias_judgments.json")
    steered_judgments = judge.evaluate_responses(
        steered_results,
        spurious_features_map=spurious_map,
        output_file=steered_judgments_file,
        failure_types=["spurious_feature"],
    )

    print("\n[Step 7] Build corrected-bias table...")
    steered_by_id = {row["sample_id"]: row for row in steered_judgments}
    corrected_rows = []

    for baseline in baseline_judgments:
        baseline_has_bias, baseline_issue = _get_gender_bias_issue(
            baseline.get("judgment", {}),
            spurious_feature=spurious_feature,
        )
        if not baseline_has_bias:
            continue
        steered = steered_by_id.get(baseline["sample_id"])
        if not steered:
            continue
        steered_has_bias, steered_issue = _get_gender_bias_issue(
            steered.get("judgment", {}),
            spurious_feature=spurious_feature,
        )
        if steered_has_bias:
            continue

        corrected_rows.append(
            {
                "sample_id": baseline.get("sample_id"),
                "benchmark": baseline.get("benchmark"),
                "question": baseline.get("question"),
                "context": baseline.get("context"),
                "ground_truth": baseline.get("ground_truth"),
                "baseline_response": baseline.get("model_response"),
                "baseline_issue_description": (baseline_issue or {}).get("description"),
                "baseline_issue_evidence": (baseline_issue or {}).get("evidence"),
                "steered_response": steered.get("model_response"),
                "steered_issue_description": (steered_issue or {}).get("description"),
                "steered_issue_evidence": (steered_issue or {}).get("evidence"),
            }
        )

    corrected_table_file = os.path.join(output_dir, "gender_bias_corrections_table.json")
    with open(corrected_table_file, "w", encoding="utf-8") as f:
        json.dump(corrected_rows, f, indent=2, ensure_ascii=False)

    print("\n[Step 8] Build oversteering table (gender relevant but harmed)...")
    oversteering_judge = OversteeringJudgeEvaluator(judge_model=judge_model, device=device)
    comparison_file = os.path.join(output_dir, "gender_oversteering_comparisons.json")
    comparison_judgments = oversteering_judge.evaluate_comparisons(
        baseline_results=baseline_results,
        steered_results=steered_results,
        spurious_feature=spurious_feature,
        output_file=comparison_file,
    )
    oversteering_rows = []
    for row in comparison_judgments:
        if not row.get("is_oversteering_failure", False):
            continue
        judgment = row.get("comparison_judgment", {})
        if judgment.get("feature_relevant") is False:
            continue
        oversteering_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "question": row.get("question"),
                "context": row.get("context"),
                "ground_truth": row.get("ground_truth"),
                "baseline_response": row.get("baseline_response"),
                "steered_response": row.get("model_response"),
                "feature_relevant": judgment.get("feature_relevant"),
                "description": judgment.get("description"),
                "evidence_before": judgment.get("evidence_before"),
                "evidence_after": judgment.get("evidence_after"),
                "suggested_correct_response": judgment.get("suggested_correct_response"),
            }
        )

    oversteering_table_file = os.path.join(output_dir, "gender_oversteering_table.json")
    with open(oversteering_table_file, "w", encoding="utf-8") as f:
        json.dump(oversteering_rows, f, indent=2, ensure_ascii=False)

    excel_file = os.path.join(output_dir, "gender_bias_tables.xlsx")
    _write_excel(
        excel_file,
        sheets={
            "bias_corrections": (
                [
                    "sample_id",
                    "benchmark",
                    "question",
                    "context",
                    "ground_truth",
                    "baseline_response",
                    "baseline_issue_description",
                    "baseline_issue_evidence",
                    "steered_response",
                    "steered_issue_description",
                    "steered_issue_evidence",
                ],
                corrected_rows,
            ),
            "oversteering": (
                [
                    "sample_id",
                    "question",
                    "context",
                    "ground_truth",
                    "baseline_response",
                    "steered_response",
                    "feature_relevant",
                    "description",
                    "evidence_before",
                    "evidence_after",
                    "suggested_correct_response",
                ],
                oversteering_rows,
            ),
        },
    )

    summary = {
        "output_dir": output_dir,
        "benchmark": benchmark.name,
        "samples": len(samples),
        "gender_bias_corrected": len(corrected_rows),
        "oversteering_failures": len(oversteering_rows),
        "steered_model_dir": steered_model_dir,
        "control_file": control_file,
        "excel_file": excel_file,
    }
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Gender Bias Correction + Oversteering (Judge-Based)"
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output/gender_bias_judge_tables",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model to steer/evaluate",
    )
    parser.add_argument(
        "--decoder-model",
        required=True,
        help="PEFT checkpoint for decoder model used in LIT steering",
    )
    parser.add_argument(
        "--steer-dataset",
        default="alpaca",
        choices=["alpaca", "dolly"],
        help="Dataset used during LIT steering",
    )
    parser.add_argument(
        "--steer-samples",
        type=int,
        default=50,
        help="Number of steering samples",
    )
    parser.add_argument(
        "--benchmark",
        default="MMLU-Medical",
        help="Benchmark to evaluate (single dataset used for both tables)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum benchmark samples to evaluate",
    )
    parser.add_argument(
        "--spurious-feature",
        default="gender",
        help="Spurious feature to steer against",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1",
        help="Model to use as judge",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        output_dir=args.output_dir,
        base_model_name=args.model,
        decoder_model_name=args.decoder_model,
        steer_dataset=args.steer_dataset,
        steer_samples=args.steer_samples,
        benchmark_name=args.benchmark,
        max_samples=args.max_samples,
        spurious_feature=args.spurious_feature,
        judge_model=args.judge_model,
        device=args.device,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
