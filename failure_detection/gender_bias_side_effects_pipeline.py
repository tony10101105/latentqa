"""
Compatibility wrapper: runs the same workflow as the former "Gender Bias + Side Effects"
pipeline by delegating to gender_bias_correction_oversteering_pipeline with run_mode=both
and bias_dataset=crows_pairs.

Outputs are written in correction_oversteering format (e.g. bias_baseline_scores.json,
bias_baseline_errors.json, gender_relevant_samples.json, gender_oversteering_*.json).
"""

import argparse
import json

from failure_detection.gender_bias_correction_oversteering_pipeline import (
    DEFAULT_GENDER_KEYWORDS,
    run_pipeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Gender Bias + Side Effects (wrapper → gender_bias_correction_oversteering)"
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output/gender_bias_side_effects",
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
        "--device",
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--crows-split",
        default="test",
        help="CrowS-Pairs split (maps to --bias-split)",
    )
    parser.add_argument(
        "--crows-max-samples",
        type=int,
        default=None,
        help="Maximum CrowS-Pairs samples (maps to --bias-max-samples)",
    )
    parser.add_argument(
        "--gender-benchmark",
        default="MMLU-Medical",
        help="Benchmark to source gender-relevant samples",
    )
    parser.add_argument(
        "--gender-max-samples",
        type=int,
        default=100,
        help="Maximum gender-relevant samples to evaluate",
    )
    parser.add_argument(
        "--gender-keywords",
        nargs="+",
        default=None,
        help="Keywords used to filter gender-relevant samples (default: built-in list)",
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
    args = parser.parse_args()

    summary = run_pipeline(
        output_dir=args.output_dir,
        base_model_name=args.model,
        decoder_model_name=args.decoder_model,
        steer_dataset=args.steer_dataset,
        steer_samples=args.steer_samples,
        device=args.device,
        bias_dataset="crows_pairs",
        bias_split=args.crows_split,
        bias_max_samples=args.crows_max_samples,
        gender_benchmark=args.gender_benchmark,
        gender_max_samples=args.gender_max_samples,
        gender_keywords=(args.gender_keywords or DEFAULT_GENDER_KEYWORDS),
        spurious_feature=args.spurious_feature,
        judge_model=args.judge_model,
        run_mode="both",
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
