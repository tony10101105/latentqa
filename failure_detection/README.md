# Failure Detection Pipeline

This module implements a pipeline to identify failure cases in LLM responses, particularly focusing on:
1. **Spurious feature usage** (e.g., using hair color in medical questions where it's irrelevant)
2. **Incorrect answers** (factually wrong or logically flawed responses)
3. **Reasoning failures** (poor reasoning or logical inconsistencies)

## Pipelines (Purpose and Overlap)

| Pipeline | Purpose |
|----------|---------|
| **pipeline** | Base failure detection (no steering): load benchmarks → run base model → LLM judge flags failures → collect failures and intervention dataset. |
| **oversteering_pipeline** | Generic oversteering: build control for any spurious feature (e.g. hair color) → steer model → run steered model on benchmarks → judge oversteering failures. |
| **debiasing_pipeline** | Debiasing only: load debiasing dataset (e.g. CrowS-Pairs) → optionally steer → score baseline and steered with logprobs → report bias metrics. No judge, no benchmark oversteering. |
| **gender_bias_side_effects_pipeline** | Compatibility wrapper: runs **gender_bias_correction_oversteering_pipeline** with `run_mode=both` and `bias_dataset=crows_pairs` (CrowS bias + oversteering on gender-relevant benchmark). Same CLI as before consolidation. |
| **paper_bias_overcorrection_pipeline** | Paper reproduction: use a **fixed** control file → CrowS-Pairs bias metrics (logprob) → steer → overcorrection evaluation on MMLU (or other benchmark) via LLM judge. |
| **gender_bias_correction_oversteering_pipeline** | **Most comprehensive** for bias + oversteering: measure bias on a debiasing dataset (baseline) → steer → report which biased samples were corrected → evaluate oversteering on gender-relevant benchmark. Supports `--run-mode bias \| oversteering \| both`. |
| **gender_bias_judge_tables_pipeline** | Single benchmark, judge-based: LLM judge finds baseline gender-biased responses → steer → report only corrected samples and oversteering failures. Outputs Excel workbook (bias_corrections + oversteering sheets) and JSON tables. |

### Overlap

- **Bias measurement (CrowS-style logprob)** appears in: `debiasing_pipeline`, `paper_bias_overcorrection_pipeline`, and `gender_bias_correction_oversteering_pipeline` (and thus in the `gender_bias_side_effects_pipeline` wrapper).
- **Oversteering on a benchmark** (baseline vs steered + judge) appears in: `oversteering_pipeline` (generic), `paper_bias_overcorrection_pipeline`, `gender_bias_correction_oversteering_pipeline`, and `gender_bias_judge_tables_pipeline`.
- **Gender-specific flow** (bias + steer + gender-relevant oversteering): `gender_bias_side_effects_pipeline` (wrapper), `gender_bias_correction_oversteering_pipeline`, and `gender_bias_judge_tables_pipeline` (which uses judge-based bias instead of logprob).
- **Steering** is used by every pipeline except the base `pipeline` and `debiasing_pipeline` (which can optionally use a pre-steered model).

### Which pipeline does the most?

- **Overall (no steering):** **pipeline** — full failure-detection flow (benchmarks → model → judge → failures → intervention dataset + categorization).
- **With steering:** **gender_bias_correction_oversteering_pipeline** — it does debiasing-dataset bias, steering, a **corrections table** (which biased samples were fixed), and full oversteering evaluation on a gender-relevant benchmark, with `--run-mode` to run bias-only, oversteering-only, or both.

---

## Overview

The main pipeline consists of four main steps:

1. **Benchmark Loading**: Load relevant benchmarks (MMLU, MedQA, TruthfulQA, etc.)
2. **Model Evaluation**: Run Meta-Llama-3-8B-Instruct on benchmark samples
3. **LLM-as-a-Judge**: Use an LLM judge to evaluate responses for failures
4. **Failure Collection**: Collect and categorize failure cases for intervention

## Quick Start

### 1. Install Dependencies

```bash
pip install datasets transformers torch fire tqdm openai
```

### 2. Run the Full Pipeline

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical MedQA \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-samples 100 \
    --output-dir failure_detection_output
```
 
### 2b. Use GPT-4.1 as Judge (Recommended)

Set your API key and pass `--judge-model gpt-4.1`:

```bash
setx OPENAI_API_KEY "your_api_key"

python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical MedQA \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --judge-model gpt-4.1 \
    --max-samples 100 \
    --output-dir failure_detection_output
```

### 3. List Available Benchmarks

```bash
python -m failure_detection.pipeline --list-benchmarks
```

## Usage Examples

### Run on Specific Benchmarks

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical PubMedQA \
    --max-samples 50 \
    --min-severity medium
```

### Only Evaluate Specific Failure Types

Use `--failure-types` to restrict judgment to specific failure modes:

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical \
    --failure-types spurious_feature \
    --max-samples 50
```

You can pass multiple types:

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical \
    --failure-types spurious_feature reasoning_failure
```

### Detect Oversteering Failures After Steering

This pipeline steers the model against a spurious feature (default: hair color),
then evaluates whether the steering caused *overcorrection* failures.

```bash
python -m failure_detection.oversteering_pipeline \
    --benchmarks MMLU-Medical \
    --spurious-feature "hair color" \
    --decoder-model path/to/decoder_checkpoint \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --judge-model gpt-4.1 \
    --max-samples 100 \
    --output-dir failure_detection_output/oversteering
```

### Use Custom Spurious Features Mapping

```bash
python -m failure_detection.pipeline \
    --benchmarks MedQA \
    --spurious-features-file failure_detection/spurious_features.json
```

### Run Individual Components

#### 1. Load Benchmarks

```python
from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name

benchmark = get_benchmark_by_name("MMLU-Medical")
samples = load_benchmark(benchmark, max_samples=100)
```

#### 2. Evaluate Model

```bash
python -m failure_detection.model_evaluator \
    --benchmark-file benchmark_samples.json \
    --output-file model_evaluations.json \
    --max-samples 100
```

#### 3. Judge Responses

```bash
python -m failure_detection.judge_evaluator \
    --evaluation-file model_evaluations.json \
    --output-file judgments.json \
    --spurious-features-file spurious_features.json \
    --judge-model gpt-4.1
```

#### 4. Collect Failures

```bash
python -m failure_detection.failure_collector \
    --judgment-file judgments.json \
    --output-file failures.json \
    --min-severity low \
    --categorize \
    --output-dir categorized_failures
```

## Output Structure

The pipeline generates the following files:

```
failure_detection_output/
├── benchmark_samples.json          # Loaded benchmark samples
├── model_evaluations.json          # Model responses
├── judgments.json                  # Judge evaluations
├── failures.json                   # Collected failure cases
├── intervention_dataset.json       # Curated dataset for intervention
└── categorized_failures/           # Failures categorized by spurious feature
    ├── spurious_feature_hair_color.json
    ├── spurious_feature_age_stereotypes.json
    └── ...
```

## Configuration

### Spurious Features

Edit `spurious_features.json` to specify which spurious features to check for each benchmark:

```json
{
  "MMLU-Medical": [
    "hair_color",
    "age_stereotypes",
    "gender_assumptions"
  ]
}
```

### Benchmarks

Available benchmarks:
- **MMLU**: Massive Multitask Language Understanding
- **MMLU-Medical**: MMLU medical subset
- **MedQA**: Medical question answering from USMLE
- **PubMedQA**: Medical QA from PubMed abstracts
- **TruthfulQA**: Truthfulness evaluation
- **HellaSwag**: Commonsense reasoning
- **CommonsenseQA**: Commonsense question answering

## Failure Types

Failures are categorized into:

1. **spurious_feature**: Model uses irrelevant features (e.g., hair color in medical questions)
2. **incorrect_answer**: Factually incorrect or logically flawed
3. **reasoning_failure**: Poor reasoning or logical inconsistencies

You can limit evaluation to a subset using `--failure-types`.

## Next Steps

After collecting failure cases:

1. **Review failures**: Examine `failures.json` to understand model weaknesses
2. **Prepare interventions**: Use `intervention_dataset.json` for decoder steering
3. **Test interventions**: Apply interventions and re-evaluate
4. **Check for hallucinations**: Test if interventions cause new failures on related tasks

## Build Decoder-Steering QA Pairs From Failures

This helper builds a `controls/*.json` file (compatible with `lit/control.py`) for a
single failure case, using deterministic QA templates based on the failure type.

### Create a QA pair for a specific failure case

```bash
python -m failure_detection.steering_qa_builder \
  --failures-file failure_detection_output/failures.json \
  --sample-id 7
```

### Choose a random failure case

```bash
python -m failure_detection.steering_qa_builder \
  --failures-file failure_detection_output/failures.json
```

The script writes a control file (e.g., `controls/failure_7.json`) that contains
QA pairs like:

- Spurious feature: “Is hair color relevant to answering this question?” → “No.”
- Incorrect answer: “Is the correct answer: <choice>?” → “Yes.”

These QA pairs provide the supervision signal for decoder steering (see `lit/control.py`).

## Notes

- The pipeline uses LLM-as-a-Judge for evaluation, which may have limitations
- Consider manual review of high-severity failures
- Adjust `min_severity` based on your needs (low/medium/high)
- GPU recommended for faster evaluation
