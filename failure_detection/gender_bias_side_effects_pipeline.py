"""
Pipeline to:
1) measure gender bias on CrowS-Pairs before steering,
2) steer the model to reduce gender bias,
3) detect side effects on samples where gender should matter.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from lit.configs.steer_config import steer_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
from lit.control import steer as lit_steer

from failure_detection.debiasing_datasets import load_crows_pairs
from failure_detection.debiasing_metrics import summarize_bias_results
from failure_detection.steering_control_builder import build_spurious_feature_control
from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name
from failure_detection.model_evaluator import ModelEvaluator
from failure_detection.steered_model_evaluator import SteeredModelEvaluator
from failure_detection.oversteering_judge import OversteeringJudgeEvaluator


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


def _score_crows_pairs(
    model,
    tokenizer,
    samples: List[Dict],
    device: str,
    max_samples: Optional[int],
) -> List[Dict]:
    if max_samples:
        samples = samples[:max_samples]
    results = []
    for item in samples:
        lp_more = _sentence_logprob(model, tokenizer, item["sent_more"], device)
        lp_less = _sentence_logprob(model, tokenizer, item["sent_less"], device)
        results.append(
            {
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
    steer_dataset: str,
    steer_samples: int,
    device: Optional[str],
    crows_split: str,
    crows_max_samples: Optional[int],
    gender_benchmark: str,
    gender_max_samples: Optional[int],
    gender_keywords: List[str],
    spurious_feature: str,
    judge_model: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("GENDER BIAS + SIDE EFFECTS PIPELINE")
    print("=" * 80)

    print("\n[Step 1] Measure gender bias on CrowS-Pairs (baseline)...")
    crows_samples = load_crows_pairs(split=crows_split)
    tokenizer = get_tokenizer(base_model_name)
    base_model = get_model(base_model_name, tokenizer, device=device)
    crows_scores = _score_crows_pairs(
        base_model,
        tokenizer,
        crows_samples,
        device=device,
        max_samples=crows_max_samples,
    )
    crows_scores_file = os.path.join(output_dir, "crows_baseline_scores.json")
    with open(crows_scores_file, "w", encoding="utf-8") as f:
        json.dump(crows_scores, f, indent=2, ensure_ascii=False)

    crows_errors = [row for row in crows_scores if row["is_bias_error"]]
    crows_errors_file = os.path.join(output_dir, "crows_baseline_bias_errors.json")
    with open(crows_errors_file, "w", encoding="utf-8") as f:
        json.dump(crows_errors, f, indent=2, ensure_ascii=False)

    crows_summary = summarize_bias_results(crows_scores)
    crows_summary_file = os.path.join(output_dir, "crows_baseline_summary.json")
    with open(crows_summary_file, "w", encoding="utf-8") as f:
        json.dump(crows_summary, f, indent=2, ensure_ascii=False)

    print(f"CrowS-Pairs baseline scores saved to {crows_scores_file}")
    print(f"CrowS-Pairs bias errors saved to {crows_errors_file}")

    print("\n[Step 2] Steer model to reduce gender bias...")
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

    print("\n[Step 3] Evaluate side effects on gender-relevant samples...")
    benchmark = get_benchmark_by_name(gender_benchmark)
    if not benchmark:
        raise ValueError(f"Benchmark not found: {gender_benchmark}")
    all_samples = load_benchmark(benchmark, max_samples=None)
    gender_samples = _filter_gender_relevant_samples(
        all_samples, keywords=gender_keywords, max_samples=gender_max_samples
    )
    gender_samples_file = os.path.join(output_dir, "gender_relevant_samples.json")
    with open(gender_samples_file, "w", encoding="utf-8") as f:
        json.dump(gender_samples, f, indent=2, ensure_ascii=False)

    baseline_eval = ModelEvaluator(model_name=base_model_name, device=device)
    baseline_eval_file = os.path.join(output_dir, "baseline_gender_relevant_evaluations.json")
    baseline_results = baseline_eval.evaluate_samples(
        gender_samples,
        output_file=baseline_eval_file,
    )

    steered_eval = SteeredModelEvaluator(
        base_model_name=base_model_name,
        peft_checkpoint=steered_model_dir,
        device=device,
    )
    steered_eval_file = os.path.join(output_dir, "steered_gender_relevant_evaluations.json")
    steered_results = steered_eval.evaluate_samples(
        gender_samples,
        output_file=steered_eval_file,
    )

    judge = OversteeringJudgeEvaluator(judge_model=judge_model, device=device)
    comparison_file = os.path.join(output_dir, "gender_oversteering_comparisons.json")
    comparison_judgments = judge.evaluate_comparisons(
        baseline_results=baseline_results,
        steered_results=steered_results,
        spurious_feature=spurious_feature,
        output_file=comparison_file,
    )
    comparison_failures = [
        j for j in comparison_judgments if j.get("is_oversteering_failure", False)
    ]
    comparison_failures_file = os.path.join(
        output_dir, "gender_oversteering_failures.json"
    )
    with open(comparison_failures_file, "w", encoding="utf-8") as f:
        json.dump(comparison_failures, f, indent=2, ensure_ascii=False)

    print(f"Gender-relevant samples saved to {gender_samples_file}")
    print(f"Baseline eval saved to {baseline_eval_file}")
    print(f"Steered eval saved to {steered_eval_file}")
    print(f"Comparison judgments saved to {comparison_file}")
    print(f"Comparison failures saved to {comparison_failures_file}")

    return {
        "output_dir": output_dir,
        "crows_baseline_summary": crows_summary,
        "crows_bias_errors": len(crows_errors),
        "gender_relevant_samples": len(gender_samples),
        "oversteering_failures": len(comparison_failures),
        "steered_model_dir": steered_model_dir,
        "control_file": control_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Gender Bias + Side Effects Pipeline")
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
        help="CrowS-Pairs split",
    )
    parser.add_argument(
        "--crows-max-samples",
        type=int,
        default=None,
        help="Maximum CrowS-Pairs samples",
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
        default=DEFAULT_GENDER_KEYWORDS,
        help="Keywords used to filter gender-relevant samples",
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
        crows_split=args.crows_split,
        crows_max_samples=args.crows_max_samples,
        gender_benchmark=args.gender_benchmark,
        gender_max_samples=args.gender_max_samples,
        gender_keywords=args.gender_keywords,
        spurious_feature=args.spurious_feature,
        judge_model=args.judge_model,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
