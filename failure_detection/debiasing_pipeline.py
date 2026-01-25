"""
Pipeline to reproduce debiasing results with LIT steering.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from lit.configs.steer_config import steer_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
from lit.control import steer as lit_steer

from failure_detection.debiasing_datasets import load_debias_dataset
from failure_detection.debiasing_metrics import summarize_bias_results
from failure_detection.steering_control_builder import build_spurious_feature_control


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


def _score_dataset(
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
            }
        )
    return results


def run_debiasing_pipeline(
    debias_dataset: str,
    spurious_feature: str,
    output_dir: str,
    base_model_name: str,
    decoder_model_name: Optional[str],
    steered_model_dir: Optional[str],
    judge_split: str,
    max_samples: Optional[int],
    device: Optional[str],
    steer_dataset: str,
    steer_samples: int,
    compute_baseline: bool,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("DEBIASING REPRODUCTION PIPELINE")
    print("=" * 80)

    control_file = build_spurious_feature_control(spurious_feature=spurious_feature)
    control_name = os.path.splitext(os.path.basename(control_file))[0]
    print(f"Control file: {control_file}")

    if steered_model_dir:
        print(f"Using existing steered model: {steered_model_dir}")
    else:
        if not decoder_model_name:
            raise ValueError("decoder_model_name is required when steered_model_dir is not provided.")
        print("\n[Step 1] Steering model...")
        steered_model_dir = _steer_model(
            control_name=control_name,
            steer_dataset=steer_dataset,
            steer_samples=steer_samples,
            target_model_name=base_model_name,
            decoder_model_name=decoder_model_name,
            device=device,
        )
        print(f"Steered model saved to {steered_model_dir}")

    print("\n[Step 2] Loading debias dataset...")
    samples = load_debias_dataset(debias_dataset, split=judge_split)
    print(f"Loaded {len(samples)} samples from {debias_dataset}")

    results = {}
    tokenizer = get_tokenizer(base_model_name)

    if compute_baseline:
        print("\n[Step 3] Scoring baseline model...")
        base_model = get_model(base_model_name, tokenizer, device=device)
        baseline_scores = _score_dataset(base_model, tokenizer, samples, device, max_samples)
        baseline_file = os.path.join(output_dir, "baseline_scores.json")
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_scores, f, indent=2, ensure_ascii=False)
        results["baseline_summary"] = summarize_bias_results(baseline_scores)
        print(f"Saved baseline scores to {baseline_file}")
    else:
        results["baseline_summary"] = None

    print("\n[Step 4] Scoring steered model...")
    steered_model = get_model(
        base_model_name,
        tokenizer,
        load_peft_checkpoint=steered_model_dir,
        device=device,
    )
    steered_scores = _score_dataset(steered_model, tokenizer, samples, device, max_samples)
    steered_file = os.path.join(output_dir, "steered_scores.json")
    with open(steered_file, "w", encoding="utf-8") as f:
        json.dump(steered_scores, f, indent=2, ensure_ascii=False)
    results["steered_summary"] = summarize_bias_results(steered_scores)
    print(f"Saved steered scores to {steered_file}")

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")

    return {
        **results,
        "output_dir": output_dir,
        "steered_model_dir": steered_model_dir,
        "control_file": control_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Debiasing Reproduction Pipeline")
    parser.add_argument(
        "--debias-dataset",
        default="crows_pairs",
        help="Debiasing dataset name (default: crows_pairs)",
    )
    parser.add_argument(
        "--spurious-feature",
        default="gender",
        help="Spurious feature to steer against (default: gender)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model to steer/evaluate",
    )
    parser.add_argument(
        "--decoder-model",
        default=None,
        help="PEFT checkpoint for decoder model used in LIT steering",
    )
    parser.add_argument(
        "--steered-model-dir",
        default=None,
        help="Existing steered model directory (skips steering step)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="failure_detection_output/debiasing",
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
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline model evaluation",
    )
    args = parser.parse_args()

    summary = run_debiasing_pipeline(
        debias_dataset=args.debias_dataset,
        spurious_feature=args.spurious_feature,
        output_dir=args.output_dir,
        base_model_name=args.model,
        decoder_model_name=args.decoder_model,
        steered_model_dir=args.steered_model_dir,
        judge_split=args.split,
        max_samples=args.max_samples,
        device=args.device,
        steer_dataset=args.steer_dataset,
        steer_samples=args.steer_samples,
        compute_baseline=not args.no_baseline,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
