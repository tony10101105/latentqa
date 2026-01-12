"""
Failure case collection and categorization script.

Collects and organizes failure cases identified by the judge for use in intervention.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import fire


class FailureCollector:
    """Collects and organizes failure cases."""
    
    def __init__(self):
        """Initialize the failure collector."""
        pass
    
    def collect_failures(
        self,
        judgment_file: str,
        output_file: str,
        min_severity: str = "low",  # Only collect failures with at least this severity
    ) -> List[Dict[str, Any]]:
        """
        Collect failure cases from judgment results.
        
        Args:
            judgment_file: Path to judgment results JSON
            output_file: Path to save collected failures
            min_severity: Minimum severity to include (low, medium, high)
        
        Returns:
            List of failure cases
        """
        print(f"Loading judgments from {judgment_file}...")
        with open(judgment_file, 'r', encoding='utf-8') as f:
            judgments = json.load(f)
        
        print(f"Loaded {len(judgments)} judgments")
        
        # Filter failures
        severity_order = {"low": 1, "medium": 2, "high": 3}
        min_severity_level = severity_order.get(min_severity, 1)
        
        failures = []
        
        for judgment in judgments:
            if not judgment.get("is_failure", False):
                continue
            
            judgment_data = judgment.get("judgment", {})
            issues = judgment_data.get("issues", [])
            
            # Check if any issue meets severity threshold
            has_valid_issue = False
            for issue in issues:
                severity = issue.get("severity", "low")
                severity_level = severity_order.get(severity, 1)
                if severity_level >= min_severity_level:
                    has_valid_issue = True
                    break
            
            if not has_valid_issue:
                continue
            
            # Create failure case entry
            failure_case = {
                "sample_id": judgment.get("sample_id"),
                "benchmark": judgment.get("benchmark"),
                "domain": judgment.get("domain"),
                "question": judgment.get("question"),
                "ground_truth": judgment.get("ground_truth"),
                "model_response": judgment.get("model_response"),
                "context": judgment.get("context"),
                "choices": judgment.get("choices"),
                "failure_type": judgment.get("failure_type"),
                "issues": issues,
                "judgment_summary": judgment_data.get("overall_assessment", ""),
            }
            
            failures.append(failure_case)
        
        print(f"Collected {len(failures)} failure cases")
        
        # Save
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        self._print_statistics(failures)
        
        return failures
    
    def _print_statistics(self, failures: List[Dict[str, Any]]):
        """Print statistics about collected failures."""
        print("\n=== Failure Statistics ===")
        
        # By benchmark
        by_benchmark = defaultdict(int)
        by_domain = defaultdict(int)
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for failure in failures:
            by_benchmark[failure["benchmark"]] += 1
            by_domain[failure["domain"]] += 1
            by_type[failure["failure_type"]] += 1
            
            for issue in failure.get("issues", []):
                severity = issue.get("severity", "unknown")
                by_severity[severity] += 1
        
        print("\nBy Benchmark:")
        for bench, count in sorted(by_benchmark.items(), key=lambda x: -x[1]):
            print(f"  {bench}: {count}")
        
        print("\nBy Domain:")
        for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
            print(f"  {domain}: {count}")
        
        print("\nBy Failure Type:")
        for ftype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {ftype}: {count}")
        
        print("\nBy Severity:")
        for severity, count in sorted(by_severity.items(), key=lambda x: -x[1]):
            print(f"  {severity}: {count}")
    
    def categorize_by_spurious_feature(
        self,
        failures_file: str,
        output_dir: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize failures by spurious feature.
        
        Args:
            failures_file: Path to failures JSON
            output_dir: Directory to save categorized failures
        
        Returns:
            Dictionary mapping spurious feature names to failure lists
        """
        print(f"Loading failures from {failures_file}...")
        with open(failures_file, 'r', encoding='utf-8') as f:
            failures = json.load(f)
        
        # Group by spurious feature
        by_feature = defaultdict(list)
        no_feature = []
        
        for failure in failures:
            if failure["failure_type"] != "spurious_feature":
                no_feature.append(failure)
                continue
            
            # Extract spurious features from issues
            features_found = set()
            for issue in failure.get("issues", []):
                if issue.get("type") == "spurious_feature":
                    feature_name = issue.get("spurious_feature_name", "unknown")
                    features_found.add(feature_name)
            
            if features_found:
                for feature in features_found:
                    by_feature[feature].append(failure)
            else:
                no_feature.append(failure)
        
        # Save categorized failures
        os.makedirs(output_dir, exist_ok=True)
        
        for feature, feature_failures in by_feature.items():
            feature_file = os.path.join(output_dir, f"spurious_feature_{feature.replace(' ', '_')}.json")
            with open(feature_file, 'w', encoding='utf-8') as f:
                json.dump(feature_failures, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(feature_failures)} failures for feature '{feature}' to {feature_file}")
        
        if no_feature:
            no_feature_file = os.path.join(output_dir, "non_spurious_failures.json")
            with open(no_feature_file, 'w', encoding='utf-8') as f:
                json.dump(no_feature, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(no_feature)} non-spurious failures to {no_feature_file}")
        
        return dict(by_feature)
    
    def create_intervention_dataset(
        self,
        failures_file: str,
        output_file: str,
        max_per_type: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create a curated dataset for intervention from failure cases.
        
        Args:
            failures_file: Path to failures JSON
            output_file: Path to save intervention dataset
            max_per_type: Maximum failures per failure type (None for all)
        
        Returns:
            List of intervention samples
        """
        print(f"Loading failures from {failures_file}...")
        with open(failures_file, 'r', encoding='utf-8') as f:
            failures = json.load(f)
        
        # Group by type
        by_type = defaultdict(list)
        for failure in failures:
            by_type[failure["failure_type"]].append(failure)
        
        # Sample from each type
        intervention_samples = []
        for ftype, type_failures in by_type.items():
            if max_per_type:
                samples = type_failures[:max_per_type]
            else:
                samples = type_failures
            
            intervention_samples.extend(samples)
        
        print(f"Created intervention dataset with {len(intervention_samples)} samples")
        
        # Save
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(intervention_samples, f, indent=2, ensure_ascii=False)
        
        return intervention_samples


def main(
    judgment_file: str,
    output_file: str,
    min_severity: str = "low",
    categorize: bool = False,
    output_dir: Optional[str] = None,
):
    """
    Main function to collect failures.
    
    Args:
        judgment_file: Path to judgment results
        output_file: Path to save collected failures
        min_severity: Minimum severity to include
        categorize: Whether to categorize by spurious feature
        output_dir: Directory for categorized outputs
    """
    collector = FailureCollector()
    
    # Collect failures
    failures = collector.collect_failures(
        judgment_file=judgment_file,
        output_file=output_file,
        min_severity=min_severity,
    )
    
    # Categorize if requested
    if categorize and output_dir:
        collector.categorize_by_spurious_feature(
            failures_file=output_file,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    fire.Fire(main)
