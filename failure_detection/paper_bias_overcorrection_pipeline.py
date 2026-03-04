"""
Reproduce paper-style bias metrics (CrowS-Pairs) and report overcorrections on MMLU.

This pipeline:
1) Uses a fixed control QA file (paper control) for steering.
2) Measures bias with CrowS-Pairs metrics (stereotype win-rate, avg logprob diff).
3) Evaluates overcorrection on MMLU medical samples via LLM judge.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lit.configs.steer_config import steer_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
from lit.control import steer as lit_steer

from failure_detection.debiasing_datasets import load_debias_dataset
from failure_detection.debiasing_metrics import summarize_bias_results
from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name
from failure_detection.model_evaluator import ModelEvaluator
from failure_detection.steered_model_evaluator import SteeredModelEvaluator
from failure_detection.oversteering_judge import OversteeringJudgeEvaluator


DEFAULT_CONTROL_FILE = os.path.join("controls", "spurious_feature_gender.json")
DEFAULT_GENDER_KEYWORDS = [
    "male",
    "female",
    "woman",
    "women",
    "man",
    "men",
    "mother",
    "father",
    "maternal",
    "paternal",
    "pregnan",
    "sex",
    "gender",
    "x-linked",
    "y-linked",
    "sex chromosome",
    "turner",
    "klinefelter",
    "androgen",
    "ovary",
    "testis",
    "uterus",
    "estrogen",
    "progesterone",
]


def _resolve_control_name(control_file: str) -> str:
    if not os.path.exists(control_file):
        raise FileNotFoundError(f"Control file not found: {control_file}")
    return os.path.splitext(os.path.basename(control_file))[0]


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


def _sentence_logprob(model, tokenizer, sentence: str, device: str) -> float:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        token_mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * token_mask
        denom = token_mask.sum().item()
    else:
        denom = token_log_probs.numel()

    return token_log_probs.sum().item() / max(denom, 1)


def _score_debias_dataset(
    model,
    tokenizer,
    samples: List[Dict],
    device: str,
    max_samples: Optional[int],
) -> List[Dict]:
    if max_samples:
        samples = samples[:max_samples]
    results = []
    for idx, item in enumerate(samples):
        lp_more = _sentence_logprob(model, tokenizer, item["sent_more"], device)
        lp_less = _sentence_logprob(model, tokenizer, item["sent_less"], device)
        results.append(
            {
                "sample_id": idx,
                "id": item.get("id"),
                "bias_type": item.get("bias_type"),
                "sent_more": item["sent_more"],
                "sent_less": item["sent_less"],
                "logprob_more": lp_more,
                "logprob_less": lp_less,
                "is_bias_error": lp_more > lp_less,
            }
        )
    return results


def _row_key(row: Dict, fallback: int) -> Tuple[str, int]:
    if row.get("id") is not None:
        return ("id", row["id"])
    return ("sample_id", row.get("sample_id", fallback))


def _filter_gender_relevant_samples(
    samples: List[Dict],
    keywords: List[str],
    max_samples: Optional[int],
) -> List[Dict]:
    pattern = re.compile(r"|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    filtered = []
    for sample in samples:
        text = " ".join(
            [
                sample.get("question", ""),
                sample.get("context", "") or "",
            ]
        )
        if pattern.search(text):
            filtered.append(sample)
    if max_samples:
        filtered = filtered[:max_samples]
    return filtered


def run_pipeline(
    output_dir: str,
    base_model_name: str,
    decoder_model_name: str,
    control_file: str,
    steer_dataset: str,
    steer_samples: int,
    device: Optional[str],
    bias_dataset: str,
    bias_split: str,
    bias_max_samples: Optional[int],
    overcorrection_benchmark: str,
    overcorrection_scope: str,
    overcorrection_max_samples: Optional[int],
    gender_keywords: List[str],
    spurious_feature: str,
    judge_model: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PAPER BIAS METRICS + MMLU OVERCORRECTION")
    print("=" * 80)

    control_name = _resolve_control_name(control_file)
    print(f"Using control file: {control_file}")

    print("\n[Step 1] Measure bias on debias dataset (baseline)...")
    debias_samples = load_debias_dataset(bias_dataset, split=bias_split)
    tokenizer = get_tokenizer(base_model_name)
    base_model = get_model(base_model_name, tokenizer, device=device)
    baseline_scores = _score_debias_dataset(
        base_model,
        tokenizer,
        debias_samples,
        device=device,
        max_samples=bias_max_samples,
    )
    baseline_scores_file = os.path.join(output_dir, "bias_baseline_scores.json")
    with open(baseline_scores_file, "w", encoding="utf-8") as f:
        json.dump(baseline_scores, f, indent=2, ensure_ascii=False)
    baseline_summary = summarize_bias_results(baseline_scores)
    baseline_summary_file = os.path.join(output_dir, "bias_baseline_summary.json")
    with open(baseline_summary_file, "w", encoding="utf-8") as f:
        json.dump(baseline_summary, f, indent=2, ensure_ascii=False)

    print("\n[Step 2] Steer model using paper control QA pairs...")
    steered_model_dir = _steer_model(
        control_name=control_name,
        steer_dataset=steer_dataset,
        steer_samples=steer_samples,
        target_model_name=base_model_name,
        decoder_model_name=decoder_model_name,
        device=device,
    )
    print(f"Steered model saved to {steered_model_dir}")

    print("\n[Step 3] Re-score debias dataset (steered) + correction table...")
    steered_model = get_model(
        base_model_name,
        tokenizer,
        load_peft_checkpoint=steered_model_dir,
        device=device,
    )
    steered_scores = _score_debias_dataset(
        steered_model,
        tokenizer,
        debias_samples,
        device=device,
        max_samples=bias_max_samples,
    )
    steered_scores_file = os.path.join(output_dir, "bias_steered_scores.json")
    with open(steered_scores_file, "w", encoding="utf-8") as f:
        json.dump(steered_scores, f, indent=2, ensure_ascii=False)
    steered_summary = summarize_bias_results(steered_scores)
    steered_summary_file = os.path.join(output_dir, "bias_steered_summary.json")
    with open(steered_summary_file, "w", encoding="utf-8") as f:
        json.dump(steered_summary, f, indent=2, ensure_ascii=False)

    baseline_bias_errors = [row for row in baseline_scores if row["is_bias_error"]]
    steered_by_key = {
        _row_key(row, i): row for i, row in enumerate(steered_scores)
    }
    corrections_table = []
    for i, baseline_row in enumerate(baseline_bias_errors):
        key = _row_key(baseline_row, i)
        steered_row = steered_by_key.get(key)
        if not steered_row:
            continue
        corrected = not steered_row.get("is_bias_error", False)
        corrections_table.append(
            {
                "id": baseline_row.get("id"),
                "sample_id": baseline_row.get("sample_id"),
                "bias_type": baseline_row.get("bias_type"),
                "sent_more": baseline_row.get("sent_more"),
                "sent_less": baseline_row.get("sent_less"),
                "baseline_logprob_more": baseline_row.get("logprob_more"),
                "baseline_logprob_less": baseline_row.get("logprob_less"),
                "baseline_is_bias_error": baseline_row.get("is_bias_error"),
                "steered_logprob_more": steered_row.get("logprob_more"),
                "steered_logprob_less": steered_row.get("logprob_less"),
                "steered_is_bias_error": steered_row.get("is_bias_error"),
                "corrected": corrected,
            }
        )
    corrections_table_file = os.path.join(output_dir, "bias_corrections_table.json")
    with open(corrections_table_file, "w", encoding="utf-8") as f:
        json.dump(corrections_table, f, indent=2, ensure_ascii=False)

    print("\n[Step 4] Evaluate overcorrection on MMLU medical...")
    benchmark = get_benchmark_by_name(overcorrection_benchmark)
    if not benchmark:
        raise ValueError(f"Benchmark not found: {overcorrection_benchmark}")
    all_samples = load_benchmark(benchmark, max_samples=None)
    if overcorrection_scope == "gender_relevant":
        overcorrection_samples = _filter_gender_relevant_samples(
            all_samples, keywords=gender_keywords, max_samples=overcorrection_max_samples
        )
    else:
        overcorrection_samples = (
            all_samples[:overcorrection_max_samples]
            if overcorrection_max_samples
            else all_samples
        )

    overcorrection_samples_file = os.path.join(
        output_dir, "overcorrection_samples.json"
    )
    with open(overcorrection_samples_file, "w", encoding="utf-8") as f:
        json.dump(overcorrection_samples, f, indent=2, ensure_ascii=False)

    baseline_eval = ModelEvaluator(model_name=base_model_name, device=device)
    baseline_eval_file = os.path.join(output_dir, "baseline_overcorrection_eval.json")
    baseline_results = baseline_eval.evaluate_samples(
        overcorrection_samples,
        output_file=baseline_eval_file,
    )

    steered_eval = SteeredModelEvaluator(
        base_model_name=base_model_name,
        peft_checkpoint=steered_model_dir,
        device=device,
    )
    steered_eval_file = os.path.join(output_dir, "steered_overcorrection_eval.json")
    steered_results = steered_eval.evaluate_samples(
        overcorrection_samples,
        output_file=steered_eval_file,
    )

    judge = OversteeringJudgeEvaluator(judge_model=judge_model, device=device)
    comparison_file = os.path.join(output_dir, "overcorrection_comparisons.json")
    comparison_judgments = judge.evaluate_comparisons(
        baseline_results=baseline_results,
        steered_results=steered_results,
        spurious_feature=spurious_feature,
        output_file=comparison_file,
    )
    overcorrection_failures = [
        j for j in comparison_judgments if j.get("is_oversteering_failure", False)
    ]
    overcorrection_failures_file = os.path.join(
        output_dir, "overcorrection_failures.json"
    )
    with open(overcorrection_failures_file, "w", encoding="utf-8") as f:
        json.dump(overcorrection_failures, f, indent=2, ensure_ascii=False)

    overcorrection_table = []
    for failure in overcorrection_failures:
        judgment = failure.get("comparison_judgment", {})
        overcorrection_table.append(
            {
                "sample_id": failure.get("sample_id"),
                "question": failure.get("question"),
                "context": failure.get("context"),
                "ground_truth": failure.get("ground_truth"),
                "baseline_response": failure.get("baseline_response"),
                "steered_response": failure.get("model_response"),
                "feature_relevant": judgment.get("feature_relevant"),
                "description": judgment.get("description"),
                "evidence_before": judgment.get("evidence_before"),
                "evidence_after": judgment.get("evidence_after"),
                "suggested_correct_response": judgment.get("suggested_correct_response"),
            }
        )
    overcorrection_table_file = os.path.join(output_dir, "overcorrection_table.json")
    with open(overcorrection_table_file, "w", encoding="utf-8") as f:
        json.dump(overcorrection_table, f, indent=2, ensure_ascii=False)

    summary = {
        "output_dir": output_dir,
        "control_file": control_file,
        "bias_dataset": bias_dataset,
        "bias_split": bias_split,
        "bias_samples": len(baseline_scores),
        "bias_errors_baseline": len(baseline_bias_errors),
        "bias_errors_corrected": sum(1 for row in corrections_table if row["corrected"]),
        "bias_baseline_summary": baseline_summary,
        "bias_steered_summary": steered_summary,
        "overcorrection_benchmark": overcorrection_benchmark,
        "overcorrection_scope": overcorrection_scope,
        "overcorrection_samples": len(overcorrection_samples),
        "overcorrection_failures": len(overcorrection_failures),
        "steered_model_dir": steered_model_dir,
    }
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Paper bias metrics + MMLU overcorrection pipeline"
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output/paper_bias_overcorrection",
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
        "--control-file",
        default=DEFAULT_CONTROL_FILE,
        help="Control QA file to use for steering",
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
        "--bias-dataset",
        default="crows_pairs",
        help="Debiasing dataset name (default: crows_pairs)",
    )
    parser.add_argument(
        "--bias-split",
        default="test",
        help="Split for debiasing dataset",
    )
    parser.add_argument(
        "--bias-max-samples",
        type=int,
        default=None,
        help="Maximum debiasing samples to score",
    )
    parser.add_argument(
        "--overcorrection-benchmark",
        default="MMLU-Medical",
        help="Benchmark to evaluate for overcorrection",
    )
    parser.add_argument(
        "--overcorrection-scope",
        default="all",
        choices=["all", "gender_relevant"],
        help="Scope of samples for overcorrection evaluation",
    )
    parser.add_argument(
        "--overcorrection-max-samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate for overcorrection",
    )
    parser.add_argument(
        "--gender-keywords",
        nargs="+",
        default=DEFAULT_GENDER_KEYWORDS,
        help="Keywords used when overcorrection-scope=gender_relevant",
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
        control_file=args.control_file,
        steer_dataset=args.steer_dataset,
        steer_samples=args.steer_samples,
        device=args.device,
        bias_dataset=args.bias_dataset,
        bias_split=args.bias_split,
        bias_max_samples=args.bias_max_samples,
        overcorrection_benchmark=args.overcorrection_benchmark,
        overcorrection_scope=args.overcorrection_scope,
        overcorrection_max_samples=args.overcorrection_max_samples,
        gender_keywords=args.gender_keywords,
        spurious_feature=args.spurious_feature,
        judge_model=args.judge_model,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
