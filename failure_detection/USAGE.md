# Usage Guide

## Running the Pipeline

### Basic Usage

```bash
python -m failure_detection.pipeline \
    --benchmarks MMLU-Medical MedQA \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-samples 100 \
    --output-dir failure_detection_output
```

**Note**: If you're using backslashes for line continuation in bash, make sure there are no spaces after the backslash, or use quotes:

```bash
python -m failure_detection.pipeline \
    --benchmarks "MMLU-Medical MedQA" \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-samples 100 \
    --output-dir failure_detection_output
```

Or use a single line:

```bash
python -m failure_detection.pipeline --benchmarks MMLU-Medical MedQA --model meta-llama/Meta-Llama-3-8B-Instruct --max-samples 100 --output-dir failure_detection_output
```

### Common Issues

1. **ModuleNotFoundError**: Make sure you're running from the project root and the `failure_detection` package is in your Python path.

2. **Dataset loading errors**: Some datasets may require authentication or may not be available. Check the error message for details.

3. **CUDA/GPU errors**: If you don't have a GPU, the model will run on CPU (slower). Make sure PyTorch is installed correctly.

### List Available Benchmarks

```bash
python -m failure_detection.pipeline --list-benchmarks
```

### Run Individual Components

You can also run individual components separately:

```bash
# 1. Load benchmarks (create a script or use Python directly)
python -c "from failure_detection.benchmarks import load_benchmark, get_benchmark_by_name; bench = get_benchmark_by_name('MMLU-Medical'); samples = load_benchmark(bench, max_samples=10); print(len(samples))"

# 2. Evaluate model
python -m failure_detection.model_evaluator \
    --benchmark-file benchmark_samples.json \
    --output-file model_evaluations.json

# 3. Judge responses
python -m failure_detection.judge_evaluator \
    --evaluation-file model_evaluations.json \
    --output-file judgments.json

# 4. Collect failures
python -m failure_detection.failure_collector \
    --judgment-file judgments.json \
    --output-file failures.json
```
