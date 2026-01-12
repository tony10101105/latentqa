"""
Model evaluation script to run Meta-Llama-3-8B-Instruct on benchmarks.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire


class ModelEvaluator:
    """Evaluates a model on benchmark datasets."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.0,  # Deterministic for evaluation
        top_p: float = 1.0,
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu), auto-detected if None
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a benchmark sample as a prompt for the model.
        
        Args:
            sample: Sample from benchmark with 'question', optionally 'context', 'choices'
        
        Returns:
            Formatted prompt string
        """
        # Use Llama-3 chat format
        messages = []
        
        # Add context if available
        if "context" in sample and sample["context"]:
            system_msg = f"Context: {sample['context']}\n\n"
        else:
            system_msg = ""
        
        # Format question
        question = sample["question"]
        
        # Add choices if multiple choice
        if "choices" in sample and sample["choices"]:
            if isinstance(sample["choices"], list):
                choices_text = "\n".join([
                    f"{chr(65+i)}. {choice}" 
                    for i, choice in enumerate(sample["choices"])
                ])
            else:
                choices_text = str(sample["choices"])
            question = f"{question}\n\nOptions:\n{choices_text}"
        
        user_msg = system_msg + question
        
        # Format as Llama-3 chat
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated response text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
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
        
        # Decode only the new tokens
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
        """
        Evaluate model on a list of samples.
        
        Args:
            samples: List of benchmark samples
            output_file: Optional file to save results
            max_samples: Maximum number of samples to evaluate
        
        Returns:
            List of evaluation results
        """
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
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
                
                # Save incrementally
                if output_file and (i + 1) % 10 == 0:
                    self._save_results(results, output_file)
            
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Final save
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main(
    benchmark_file: str,
    output_file: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Main evaluation function.
    
    Args:
        benchmark_file: Path to JSON file with benchmark samples
        output_file: Path to save evaluation results
        model_name: Model to evaluate
        max_samples: Maximum samples to evaluate (None for all)
        device: Device to use (cuda/cpu)
    """
    # Load samples
    print(f"Loading samples from {benchmark_file}...")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_name=model_name, device=device)
    
    # Evaluate
    results = evaluator.evaluate_samples(
        samples,
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\nEvaluation complete! Results saved to {output_file}")
    print(f"Evaluated {len(results)} samples")


if __name__ == "__main__":
    fire.Fire(main)
