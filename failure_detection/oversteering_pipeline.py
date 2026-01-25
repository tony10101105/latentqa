"""
Pipeline to detect oversteering failures after spurious-feature steering.
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional

from lit.configs.steer_config import steer_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
from lit.control import steer as lit_steer

from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name
from failure_detection.steering_control_builder import build_spurious_feature_control
from failure_detection.steered_model_evaluator import SteeredModelEvaluator
from failure_detection.oversteering_judge import OversteeringJudgeEvaluator


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


def run_oversteering_pipeline(
    benchmark_names: List[str],
    spurious_feature: str,
    max_samples_per_benchmark: Optional[int],
    output_dir: str,
    target_model_name: str,
    decoder_model_name: str,
    judge_model: str,
    steer_dataset: str,
    steer_samples: int,
    device: Optional[str],
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("OVERSTEERING FAILURE PIPELINE")
    print("=" * 80)

    control_file = build_spurious_feature_control(spurious_feature=spurious_feature)
    control_name = os.path.splitext(os.path.basename(control_file))[0]
    print(f"Control file: {control_file}")

    print("\n[Step 1] Steering model...")
    steered_model_dir = _steer_model(
        control_name=control_name,
        steer_dataset=steer_dataset,
        steer_samples=steer_samples,
        target_model_name=target_model_name,
        decoder_model_name=decoder_model_name,
        device=device or "cuda:0",
    )
    print(f"Steered model saved to {steered_model_dir}")

    print("\n[Step 2] Loading benchmarks...")
    all_samples = []
    benchmark_info = {}
    for bench_name in benchmark_names:
        benchmark = get_benchmark_by_name(bench_name)
        if not benchmark:
            print(f"Warning: Benchmark '{bench_name}' not found. Skipping.")
            continue
        print(f"  Loading {benchmark.name}...")
        samples = load_benchmark(benchmark, max_samples=max_samples_per_benchmark)
        if samples:
            all_samples.extend(samples)
            benchmark_info[bench_name] = {
                "samples": len(samples),
                "domain": benchmark.domain,
                "spurious_features": benchmark.spurious_features or [],
            }
            print(f"    Loaded {len(samples)} samples")
        else:
            print(f"    Warning: No samples loaded for {bench_name}")
    if not all_samples:
        raise ValueError("No samples loaded from any benchmark!")

    samples_file = os.path.join(output_dir, "benchmark_samples.json")
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved samples to {samples_file}")

    print("\n[Step 3] Evaluating steered model...")
    evaluator = SteeredModelEvaluator(
        base_model_name=target_model_name,
        peft_checkpoint=steered_model_dir,
        device=device,
    )
    evaluation_file = os.path.join(output_dir, "steered_model_evaluations.json")
    evaluation_results = evaluator.evaluate_samples(
        all_samples,
        output_file=evaluation_file,
    )
    print(f"Evaluation complete! Results saved to {evaluation_file}")

    print("\n[Step 4] Judging oversteering failures...")
    judge = OversteeringJudgeEvaluator(judge_model=judge_model, device=device)
    judgments_file = os.path.join(output_dir, "oversteering_judgments.json")
    judgments = judge.evaluate_responses(
        evaluation_results,
        spurious_feature=spurious_feature,
        output_file=judgments_file,
    )
    failures = [j for j in judgments if j.get("is_oversteering_failure", False)]
    failures_file = os.path.join(output_dir, "oversteering_failures.json")
    with open(failures_file, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Generated files:")
    print(f"  - {samples_file}")
    print(f"  - {evaluation_file}")
    print(f"  - {judgments_file}")
    print(f"  - {failures_file}")

    return {
        "total_samples": len(all_samples),
        "total_evaluations": len(evaluation_results),
        "total_judgments": len(judgments),
        "oversteering_failures": len(failures),
        "benchmark_info": benchmark_info,
        "output_dir": output_dir,
        "steered_model_dir": steered_model_dir,
        "control_file": control_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Oversteering Failure Pipeline")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["MMLU-Medical"],
        help="Benchmark names to evaluate",
    )
    parser.add_argument(
        "--spurious-feature",
        default="hair color",
        help="Spurious feature to steer against",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Target model to steer and evaluate",
    )
    parser.add_argument(
        "--decoder-model",
        required=True,
        help="PEFT checkpoint for the decoder model used in LIT steering",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1",
        help="Model to use as judge",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output/oversteering",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu)",
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
    args = parser.parse_args()

    summary = run_oversteering_pipeline(
        benchmark_names=args.benchmarks,
        spurious_feature=args.spurious_feature,
        max_samples_per_benchmark=args.max_samples,
        output_dir=args.output_dir,
        target_model_name=args.model,
        decoder_model_name=args.decoder_model,
        judge_model=args.judge_model,
        steer_dataset=args.steer_dataset,
        steer_samples=args.steer_samples,
        device=args.device,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
