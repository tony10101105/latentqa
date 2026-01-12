# Failure Detection Pipeline - Summary

## Overview

This pipeline identifies failure cases in Meta-Llama-3-8B-Instruct where the model:
1. **Uses spurious features** (e.g., hair color in medical questions where it's irrelevant)
2. **Gives incorrect answers** (factually wrong or logically flawed)
3. **Shows reasoning failures** (poor logical reasoning)

These failure cases are collected for use in intervention experiments (decoder steering) to fix the model's behavior.

## Pipeline Architecture

```
┌─────────────────┐
│ 1. Load         │  Load benchmarks (MMLU, MedQA, etc.)
│   Benchmarks    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Evaluate     │  Run Meta-Llama-3-8B-Instruct on samples
│   Model         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Judge        │  Use LLM-as-a-Judge to identify failures
│   Responses     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Collect      │  Collect and categorize failure cases
│   Failures     │
└─────────────────┘
```

## Key Components

### 1. `benchmarks.py`
- Identifies and loads relevant benchmarks
- Supports: MMLU, MedQA, PubMedQA, TruthfulQA, HellaSwag, CommonsenseQA
- Handles multiple choice and open-ended questions
- Tracks spurious features per benchmark

### 2. `model_evaluator.py`
- Runs Meta-Llama-3-8B-Instruct on benchmark samples
- Formats prompts using Llama-3 chat format
- Handles multiple choice and context-based questions
- Saves evaluation results

### 3. `judge_evaluator.py`
- Uses LLM-as-a-Judge to evaluate responses
- Checks for:
  - Spurious feature usage (e.g., hair color, demographics)
  - Incorrect answers
  - Reasoning failures
- Returns structured judgments with severity levels

### 4. `failure_collector.py`
- Collects failure cases from judgments
- Categorizes by failure type and spurious feature
- Creates intervention datasets
- Provides statistics and summaries

### 5. `pipeline.py`
- Orchestrates the full pipeline
- Handles command-line interface
- Manages output files and directories

## Quick Start

### Installation

```bash
cd failure_detection
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical MedQA \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-samples 100 \
    --output-dir failure_detection_output
```

### List Available Benchmarks

```bash
python -m failure_detection.pipeline --list-benchmarks
```

## Example: Finding Hair Color Spurious Feature

To find cases where the model uses hair color inappropriately in medical questions:

1. **Run pipeline on medical benchmarks:**
```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical MedQA \
    --max-samples 200 \
    --spurious-features-file failure_detection/spurious_features.json
```

2. **Check categorized failures:**
```bash
cat failure_detection_output/categorized_failures/spurious_feature_hair_color.json
```

3. **Review intervention dataset:**
```bash
cat failure_detection_output/intervention_dataset.json
```

## Output Files

The pipeline generates:

- `benchmark_samples.json`: Loaded benchmark samples
- `model_evaluations.json`: Model responses to each question
- `judgments.json`: Judge evaluations with failure flags
- `failures.json`: Collected failure cases
- `intervention_dataset.json`: Curated dataset for intervention
- `categorized_failures/`: Failures grouped by spurious feature

## Configuration

### Spurious Features

Edit `spurious_features.json` to specify which features to check:

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
- **MMLU-Medical**: Medical subset of MMLU (good for spurious features)
- **MedQA**: USMLE medical questions
- **PubMedQA**: Medical QA from PubMed
- **TruthfulQA**: Truthfulness evaluation
- **MMLU**: Full MMLU benchmark
- **HellaSwag**: Commonsense reasoning
- **CommonsenseQA**: Commonsense QA

## Failure Types

1. **spurious_feature**: Model uses irrelevant features
   - Example: Mentions hair color when diagnosing a medical condition
   - Severity: low/medium/high

2. **incorrect_answer**: Factually wrong
   - Example: Gives incorrect medical diagnosis
   - Severity: low/medium/high

3. **reasoning_failure**: Poor reasoning
   - Example: Logical inconsistencies in explanation
   - Severity: low/medium/high

## Next Steps After Collection

1. **Review failures**: Examine `failures.json` to understand patterns
2. **Prepare interventions**: Use failure cases to create decoder steering controls
3. **Test interventions**: Apply interventions and re-evaluate
4. **Check side effects**: Test if interventions cause hallucinations on related tasks

## Example Workflow

```bash
# Step 1: Run pipeline
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical \
    --max-samples 50 \
    --output-dir output

# Step 2: Review failures
python -c "
import json
with open('output/failures.json') as f:
    failures = json.load(f)
print(f'Found {len(failures)} failures')
for f in failures[:3]:
    print(f\"\\nQuestion: {f['question'][:100]}...\")
    print(f\"Failure: {f['failure_type']}\")
"

# Step 3: Use for intervention
# The intervention_dataset.json can be used with decoder steering
```

## Notes

- **GPU recommended** for faster evaluation
- **LLM-as-a-Judge** may have limitations; consider manual review for critical cases
- **Adjust severity** based on needs (low catches more, high catches only severe issues)
- **Max samples** can be set per benchmark to control evaluation time

## Integration with Intervention

After collecting failures, you can:

1. Use `intervention_dataset.json` to identify what needs fixing
2. Create decoder steering controls based on failure patterns
3. Test interventions on the same benchmarks
4. Check for hallucinations on related tasks (e.g., if you fix hair color usage, test if model still uses it correctly when it IS relevant)

## Example: Hair Color Intervention

If you find the model uses hair color inappropriately:

1. **Before intervention**: Model mentions hair color in medical diagnosis
2. **Apply intervention**: Use decoder steering to reduce hair color relevance
3. **Test on original task**: Should no longer use hair color inappropriately
4. **Test on related task**: If asked about hair color explicitly, should still respond correctly (no hallucination)

This pipeline helps you find the "before intervention" cases automatically!
