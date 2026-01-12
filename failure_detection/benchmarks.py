"""
Benchmark identification and loading module.

This module identifies and loads relevant benchmarks for testing model failures,
particularly focusing on spurious correlations and incorrect answers.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not installed. Install with: pip install datasets")
    load_dataset = None


@dataclass
class Benchmark:
    """Represents a benchmark dataset."""
    name: str
    description: str
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "test"
    question_field: str = "question"
    answer_field: str = "answer"
    context_field: Optional[str] = None
    multiple_choice: bool = False
    choices_field: Optional[str] = None
    spurious_features: List[str] = None  # Known spurious features to check for
    domain: str = "general"  # e.g., "medical", "reasoning", "general"


# Curated list of benchmarks suitable for failure detection
BENCHMARKS = [
    Benchmark(
        name="MMLU",
        description="Massive Multitask Language Understanding - 15,908 multiple-choice questions across 57 subjects",
        dataset_path="cais/mmlu",
        split="test",
        question_field="question",
        answer_field="answer",
        multiple_choice=True,
        choices_field="choices",
        domain="general",
        spurious_features=["demographic_biases", "cultural_assumptions"]
    ),
    Benchmark(
        name="MMLU-Medical",
        description="MMLU medical subset for medical question answering",
        dataset_path="cais/mmlu",
        dataset_config="medical_genetics",
        split="test",
        question_field="question",
        answer_field="answer",
        multiple_choice=True,
        choices_field="choices",
        domain="medical",
        spurious_features=["demographic_biases", "hair_color", "age_stereotypes"]
    ),
    Benchmark(
        name="MedQA",
        description="Medical question answering from USMLE exams",
        dataset_path="bigbio/med_qa",
        dataset_config="med_qa_en",
        split="test",
        question_field="question",
        answer_field="answer",
        multiple_choice=True,
        domain="medical",
        spurious_features=["demographic_biases", "hair_color", "age_stereotypes", "gender_assumptions"]
    ),
    Benchmark(
        name="PubMedQA",
        description="Medical question answering from PubMed abstracts",
        dataset_path="pubmed_qa",
        dataset_config="pqa_labeled",
        split="train",  # Note: test split may be limited
        question_field="question",
        answer_field="final_decision",
        context_field="context",
        domain="medical",
        spurious_features=["demographic_biases", "hair_color"]
    ),
    Benchmark(
        name="TruthfulQA",
        description="817 questions designed to test truthfulness and identify misconceptions",
        dataset_path="truthful_qa",
        dataset_config="generation",
        split="validation",
        question_field="Question",
        answer_field="Best Answer",
        domain="general",
        spurious_features=["common_misconceptions", "spurious_correlations"]
    ),
    Benchmark(
        name="HellaSwag",
        description="Commonsense reasoning benchmark",
        dataset_path="Rowan/hellaswag",
        split="validation",
        question_field="ctx",
        answer_field="label",
        multiple_choice=True,
        choices_field="endings",
        domain="reasoning",
        spurious_features=["superficial_patterns"]
    ),
    Benchmark(
        name="CommonsenseQA",
        description="Commonsense question answering",
        dataset_path="commonsense_qa",
        split="validation",
        question_field="question",
        answer_field="answerKey",
        multiple_choice=True,
        choices_field="choices",
        domain="reasoning",
        spurious_features=["superficial_patterns"]
    ),
]


def load_benchmark(benchmark: Benchmark, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load a benchmark dataset.
    
    Args:
        benchmark: Benchmark configuration
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of samples from the benchmark
    """
    if load_dataset is None:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    try:
        print(f"  Loading dataset: {benchmark.dataset_path}" + 
              (f" (config: {benchmark.dataset_config})" if benchmark.dataset_config else ""))
        
        if benchmark.dataset_config:
            dataset = load_dataset(
                benchmark.dataset_path,
                benchmark.dataset_config,
                split=benchmark.split
            )
        else:
            dataset = load_dataset(benchmark.dataset_path, split=benchmark.split)
        
        # Convert to list and limit samples
        samples = list(dataset)
        print(f"  Loaded {len(samples)} samples from dataset")
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"  Limited to {len(samples)} samples")
        
        # Normalize field names
        normalized_samples = []
        for sample in samples:
            normalized = {
                "question": sample.get(benchmark.question_field, ""),
                "answer": sample.get(benchmark.answer_field, ""),
                "benchmark": benchmark.name,
                "domain": benchmark.domain,
            }
            
            if benchmark.context_field and benchmark.context_field in sample:
                normalized["context"] = sample[benchmark.context_field]
            
            if benchmark.multiple_choice:
                if benchmark.choices_field and benchmark.choices_field in sample:
                    normalized["choices"] = sample[benchmark.choices_field]
            
            # Store original sample for reference
            normalized["original"] = sample
            
            normalized_samples.append(normalized)
        
        return normalized_samples
    
    except Exception as e:
        import traceback
        print(f"Error loading benchmark {benchmark.name}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return []


def load_custom_benchmark(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a custom benchmark from a JSON file.
    
    Expected format:
    [
        {
            "question": "...",
            "answer": "...",
            "context": "...",  # optional
            "benchmark": "...",
            "domain": "...",
            ...
        }
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_benchmark_by_name(name: str) -> Optional[Benchmark]:
    """Get a benchmark configuration by name."""
    for bench in BENCHMARKS:
        if bench.name.lower() == name.lower():
            return bench
    return None


def list_available_benchmarks() -> List[str]:
    """List all available benchmark names."""
    return [bench.name for bench in BENCHMARKS]


if __name__ == "__main__":
    # Example usage
    print("Available benchmarks:")
    for bench in BENCHMARKS:
        print(f"  - {bench.name}: {bench.description}")
    
    # Test loading a small sample
    if load_dataset:
        mmlu_medical = get_benchmark_by_name("MMLU-Medical")
        if mmlu_medical:
            print(f"\nLoading sample from {mmlu_medical.name}...")
            samples = load_benchmark(mmlu_medical, max_samples=5)
            print(f"Loaded {len(samples)} samples")
            if samples:
                print(f"\nExample sample:")
                print(json.dumps(samples[0], indent=2))
