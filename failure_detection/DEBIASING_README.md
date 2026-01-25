# Debiasing Reproduction Pipeline

This document explains how to reproduce the debiasing experiments using the
separate debiasing pipeline. The pipeline is isolated from the standard failure
detection and oversteering pipelines.

## What It Does

1. **Builds a control** that steers against a spurious feature (default: gender).
2. **Runs LIT steering** to produce a PEFT adapter.
3. **Scores bias** on a debiasing dataset by comparing log-likelihoods of
   stereotype vs anti-stereotype sentence pairs.
4. **Writes summaries** comparing baseline vs steered model performance.

## Recommended Dataset to Start

Start with **CrowS-Pairs** (`crows_pairs`). It is small, widely used, and the
metric is simple: how often the model assigns higher likelihood to the
stereotyped sentence than the anti-stereotyped one.

## Example Command (Gender Debiasing)

```
python -m failure_detection.debiasing_pipeline \
  --debias-dataset crows_pairs \
  --spurious-feature "gender" \
  --decoder-model aypan17/latentqa_llama-3-8b-instruct \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --max-samples 100 \
  --output-dir failure_detection_output/debiasing_gender
```

## Using an Existing Steered Model

If you already have a steered adapter from a previous run:

```
python -m failure_detection.debiasing_pipeline \
  --debias-dataset crows_pairs \
  --spurious-feature "gender" \
  --steered-model-dir out/model/steer_spurious_feature_gender_alpaca_50 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --max-samples 100 \
  --output-dir failure_detection_output/debiasing_gender
```

## Outputs

Written to `--output-dir`:

- `baseline_scores.json`: Per-example log-likelihoods for the base model.
- `steered_scores.json`: Per-example log-likelihoods for the steered model.
- `summary.json`: Aggregate metrics (stereotype win-rate and average logprob diff).

## Metrics

For each pair:

- `logprob_more`: log-likelihood of the stereotyped sentence.
- `logprob_less`: log-likelihood of the anti-stereotyped sentence.

Aggregates in `summary.json`:

- `stereotype_win_rate`: fraction of pairs where `logprob_more > logprob_less`.
- `avg_logprob_diff`: average of `logprob_more - logprob_less`.

Lower is better (less bias).

## Notes

- `--decoder-model` must reference a valid PEFT adapter repo or local path
  (containing `adapter_config.json`).
- GPU is strongly recommended.
- This pipeline evaluates *likelihood bias* and does not use the LLM judge.
