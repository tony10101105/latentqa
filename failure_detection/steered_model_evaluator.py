"""
Evaluate a PEFT-steered model on benchmark samples.
"""

import json
import os
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
import fire

from lit.utils.infra_utils import get_model, get_tokenizer


class SteeredModelEvaluator:
    """Evaluates a PEFT-steered model on benchmark datasets."""

    def __init__(
        self,
        base_model_name: str,
        peft_checkpoint: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.base_model_name = base_model_name
        self.peft_checkpoint = peft_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading steered model {base_model_name} with PEFT from {peft_checkpoint} on {self.device}...")
        self.tokenizer = get_tokenizer(base_model_name)
        self.model = get_model(
            model_name=base_model_name,
            tokenizer=self.tokenizer,
            load_peft_checkpoint=peft_checkpoint,
            device=self.device,
        )
        print("Steered model loaded successfully!")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        messages = []

        if "context" in sample and sample["context"]:
            system_msg = f"Context: {sample['context']}\n\n"
        else:
            system_msg = ""

        question = sample["question"]

        if "choices" in sample and sample["choices"]:
            if isinstance(sample["choices"], list):
                choices_text = "\n".join(
                    [f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample["choices"])]
                )
            else:
                choices_text = str(sample["choices"])
            question = f"{question}\n\nOptions:\n{choices_text}"

        user_msg = system_msg + question

        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def evaluate_samples(
        self,
        samples: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if max_samples:
            samples = samples[:max_samples]

        results = []

        for i, sample in enumerate(tqdm(samples, desc="Evaluating steered model")):
            try:
                prompt = self.format_prompt(sample)
                response = self.generate_response(prompt)

                result = {
                    "sample_id": i,
                    "benchmark": sample.get("benchmark", "unknown"),
                    "domain": sample.get("domain", "general"),
                    "question": sample["question"],
                    "ground_truth": sample.get("answer", ""),
                    "model_response": response,
                    "context": sample.get("context", ""),
                    "choices": sample.get("choices", []),
                }
                results.append(result)

                if output_file and (i + 1) % 10 == 0:
                    self._save_results(results, output_file)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        if output_file:
            self._save_results(results, output_file)
        return results

    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main(
    benchmark_file: str,
    output_file: str,
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    peft_checkpoint: str = "",
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
):
    print(f"Loading samples from {benchmark_file}...")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples")
    evaluator = SteeredModelEvaluator(
        base_model_name=base_model_name,
        peft_checkpoint=peft_checkpoint,
        device=device,
    )
    results = evaluator.evaluate_samples(
        samples,
        output_file=output_file,
        max_samples=max_samples,
    )
    print(f"\nEvaluation complete! Results saved to {output_file}")
    print(f"Evaluated {len(results)} samples")


if __name__ == "__main__":
    fire.Fire(main)
