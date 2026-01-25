# Oversteering Failure Pipeline

This document explains how the oversteering pipeline works and how to use it.
The goal is to detect *overcorrection failures* caused by steering a model
away from a spurious feature (e.g., hair color) and then evaluating whether the
steered model incorrectly dismisses that feature when it is actually relevant.

## What the Pipeline Does

The pipeline performs four stages:

1. **Build control file** for a spurious feature.
2. **Steer the model** using LIT to create a PEFT adapter.
3. **Evaluate the steered model** on benchmark samples.
4. **Judge oversteering failures** with an LLM judge.

The output is a filtered list of samples where the steering intervention appears
to have caused a wrong answer specifically due to the spurious‑feature correction.

## Inputs and Outputs

### Inputs

- `--spurious-feature`: The feature to suppress during steering (default: `hair color`).
- `--decoder-model`: PEFT checkpoint used by LIT for steering (local path or HF repo).
- `--model`: Base model to steer/evaluate.
- `--benchmarks`: Benchmarks to evaluate.
- `--judge-model`: LLM used to judge oversteering failures.

### Outputs

Written to `--output-dir`:

- `benchmark_samples.json`: Loaded samples.
- `steered_model_evaluations.json`: Model responses after steering.
- `oversteering_judgments.json`: Judge outputs per sample.
- `oversteering_failures.json`: Only the oversteering failures.

The steered PEFT adapter is saved under:

- `out/model/steer_<control>_<dataset>_<samples>/`

## How Each Stage Works

### 1) Control File Generation

`failure_detection/steering_control_builder.py` creates a control JSON for the
spurious feature. Example control entries:

- “Is hair color relevant to answering this question?” → “No.”
- “Does the answer depend on hair color?” → “No.”

The file is stored in `controls/` and then used by LIT during steering.

### 2) LIT Steering

`failure_detection/oversteering_pipeline.py` invokes LIT steering using:

- `lit/control.py` for training the adapter
- `lit/configs/steer_config.py` for default steering hyperparameters

The result is a PEFT adapter saved to `out/model/steer_*`.

### 3) Steered Model Evaluation

`failure_detection/steered_model_evaluator.py` loads:

- Base model (`--model`)
- PEFT adapter (`out/model/steer_*`)

It then evaluates the steered model on benchmark samples and saves the results
to `steered_model_evaluations.json`.

### 4) Oversteering Judgment

`failure_detection/oversteering_judge.py` asks a judge model:

> Is the response wrong because the model dismissed the steered spurious feature
> when it was actually relevant?

Only those failures are retained in `oversteering_failures.json`.

## Example Usage

```bash
python -m failure_detection.oversteering_pipeline \
  --benchmarks MMLU-Medical \
  --spurious-feature "hair color" \
  --decoder-model aypan17/latentqa_llama-3-8b-instruct \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --judge-model gpt-4.1 \
  --max-samples 100 \
  --output-dir failure_detection_output/oversteering
```

## Notes and Tips

- **Decoder model required**: `--decoder-model` must point to a PEFT adapter
  (contains `adapter_config.json` and adapter weights).
- **GPU strongly recommended** for steering and evaluation.
- **Judge cost**: Using `gpt-4.1` as judge may incur API costs.
- **Interpreting failures**: Oversteering failures represent cases where the
  spurious feature is actually relevant, but the steered model rejects it.

## Files to Inspect for Analysis

- `oversteering_failures.json`: Primary list of oversteering errors.
- `oversteering_judgments.json`: Full judgments for all evaluated samples.
- `steered_model_evaluations.json`: Raw model outputs before judging.
