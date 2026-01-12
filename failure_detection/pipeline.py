"""
Main pipeline orchestrator for failure detection.

This script orchestrates the full pipeline:
1. Load benchmarks
2. Evaluate model on benchmarks
3. Judge responses for failures
4. Collect failure cases
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    # Try relative imports first (when run as module)
    from .benchmarks import (
        BENCHMARKS,
        load_benchmark,
        get_benchmark_by_name,
        list_available_benchmarks,
    )
    from .model_evaluator import ModelEvaluator
    from .judge_evaluator import JudgeEvaluator
    from .failure_collector import FailureCollector
except ImportError:
    # Fall back to absolute imports (when run directly)
    from failure_detection.benchmarks import (
        BENCHMARKS,
        load_benchmark,
        get_benchmark_by_name,
        list_available_benchmarks,
    )
    from failure_detection.model_evaluator import ModelEvaluator
    from failure_detection.judge_evaluator import JudgeEvaluator
    from failure_detection.failure_collector import FailureCollector


class FailureDetectionPipeline:
    """Orchestrates the failure detection pipeline."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        judge_model: Optional[str] = None,
        device: Optional[str] = None,
        output_dir: str = "failure_detection_output",
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_name: Model to evaluate
            judge_model: Model to use as judge (defaults to same as model_name)
            device: Device to use
            output_dir: Directory for outputs
        """
        self.model_name = model_name
        self.judge_model = judge_model or model_name
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_full_pipeline(
        self,
        benchmark_names: List[str],
        max_samples_per_benchmark: Optional[int] = None,
        min_severity: str = "low",
        spurious_features_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full failure detection pipeline.
        
        Args:
            benchmark_names: List of benchmark names to evaluate
            max_samples_per_benchmark: Max samples per benchmark
            min_severity: Minimum severity for failure collection
            spurious_features_file: Path to spurious features mapping
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        print("=" * 80)
        print("FAILURE DETECTION PIPELINE")
        print("=" * 80)
        
        # Step 1: Load benchmarks
        print("\n[Step 1] Loading benchmarks...")
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
        
        print(f"\nTotal samples loaded: {len(all_samples)}")
        
        # Save loaded samples
        samples_file = os.path.join(self.output_dir, "benchmark_samples.json")
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"Saved samples to {samples_file}")
        
        # Step 2: Evaluate model
        print("\n[Step 2] Evaluating model on benchmarks...")
        evaluator = ModelEvaluator(model_name=self.model_name, device=self.device)
        
        evaluation_file = os.path.join(self.output_dir, "model_evaluations.json")
        evaluation_results = evaluator.evaluate_samples(
            all_samples,
            output_file=evaluation_file,
        )
        
        print(f"Evaluation complete! Results saved to {evaluation_file}")
        
        # Step 3: Judge responses
        print("\n[Step 3] Judging responses for failures...")
        
        # Load spurious features map
        spurious_features_map = {}
        if spurious_features_file and os.path.exists(spurious_features_file):
            with open(spurious_features_file, 'r', encoding='utf-8') as f:
                spurious_features_map = json.load(f)
        else:
            # Create from benchmark info
            for bench_name, info in benchmark_info.items():
                if info["spurious_features"]:
                    spurious_features_map[bench_name] = info["spurious_features"]
        
        judge = JudgeEvaluator(judge_model=self.judge_model, device=self.device)
        
        judgment_file = os.path.join(self.output_dir, "judgments.json")
        judgments = judge.evaluate_responses(
            evaluation_results,
            spurious_features_map=spurious_features_map,
            output_file=judgment_file,
        )
        
        # Statistics
        failures = [j for j in judgments if j.get("is_failure", False)]
        print(f"Judgment complete!")
        print(f"  Total responses: {len(judgments)}")
        print(f"  Failure cases: {len(failures)} ({100*len(failures)/len(judgments):.1f}%)")
        
        # Step 4: Collect failures
        print("\n[Step 4] Collecting failure cases...")
        collector = FailureCollector()
        
        failures_file = os.path.join(self.output_dir, "failures.json")
        collected_failures = collector.collect_failures(
            judgment_file=judgment_file,
            output_file=failures_file,
            min_severity=min_severity,
        )
        
        # Categorize by spurious feature
        categorized_dir = os.path.join(self.output_dir, "categorized_failures")
        collector.categorize_by_spurious_feature(
            failures_file=failures_file,
            output_dir=categorized_dir,
        )
        
        # Create intervention dataset
        intervention_file = os.path.join(self.output_dir, "intervention_dataset.json")
        intervention_samples = collector.create_intervention_dataset(
            failures_file=failures_file,
            output_file=intervention_file,
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - {samples_file}")
        print(f"  - {evaluation_file}")
        print(f"  - {judgment_file}")
        print(f"  - {failures_file}")
        print(f"  - {intervention_file}")
        print(f"  - {categorized_dir}/ (categorized failures)")
        
        # Return summary
        return {
            "total_samples": len(all_samples),
            "total_evaluations": len(evaluation_results),
            "total_judgments": len(judgments),
            "failure_cases": len(collected_failures),
            "intervention_samples": len(intervention_samples),
            "benchmark_info": benchmark_info,
            "output_dir": self.output_dir,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Failure Detection Pipeline")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["MMLU-Medical", "MedQA"],
        help="Benchmark names to evaluate",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model to use as judge (defaults to same as --model)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark",
    )
    parser.add_argument(
        "--min-severity",
        default="low",
        choices=["low", "medium", "high"],
        help="Minimum severity for failure collection",
    )
    parser.add_argument(
        "--spurious-features-file",
        default=None,
        help="JSON file mapping benchmarks to spurious features",
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_benchmarks:
        print("Available benchmarks:")
        for bench in BENCHMARKS:
            print(f"  - {bench.name}: {bench.description}")
            print(f"    Domain: {bench.domain}")
            if bench.spurious_features:
                print(f"    Spurious features: {', '.join(bench.spurious_features)}")
        return
    
    # Run pipeline
    pipeline = FailureDetectionPipeline(
        model_name=args.model,
        judge_model=args.judge_model,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    summary = pipeline.run_full_pipeline(
        benchmark_names=args.benchmarks,
        max_samples_per_benchmark=args.max_samples,
        min_severity=args.min_severity,
        spurious_features_file=args.spurious_features_file,
    )
    
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
