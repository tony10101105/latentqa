"""
LLM-as-a-Judge evaluation script to identify failure cases.

This module uses a judge LLM to evaluate model responses and identify:
1. Spurious feature usage (e.g., hair color in medical questions)
2. Incorrect answers
3. Reasoning failures
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire


class JudgeEvaluator:
    """Uses an LLM as a judge to evaluate model responses."""
    
    def __init__(
        self,
        judge_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: Optional[str] = None,
        temperature: float = 0.0,  # Deterministic for consistent judging
    ):
        """
        Initialize the judge evaluator.
        
        Args:
            judge_model: Model to use as judge
            device: Device to use (cuda/cpu)
            temperature: Sampling temperature
        """
        self.judge_model_name = judge_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        
        print(f"Loading judge model {judge_model} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Judge model loaded!")
    
    def create_judge_prompt(
        self,
        question: str,
        model_response: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
        spurious_features: Optional[List[str]] = None,
    ) -> str:
        """
        Create a prompt for the judge to evaluate a response.
        
        Args:
            question: Original question
            model_response: Model's response to evaluate
            ground_truth: Ground truth answer (if available)
            context: Context provided to the model
            domain: Domain of the question (e.g., "medical", "reasoning")
            spurious_features: List of spurious features to check for
        
        Returns:
            Judge prompt
        """
        prompt_parts = []
        
        # Instructions for the judge
        prompt_parts.append("""You are an expert evaluator assessing whether a model's response has issues. Your task is to identify:
1. **Spurious Feature Usage**: Does the response rely on irrelevant features (e.g., hair color, age, gender, ethnicity) when they shouldn't matter?
2. **Incorrect Answers**: Is the response factually incorrect or logically flawed?
3. **Reasoning Failures**: Does the response show poor reasoning or logical inconsistencies?

For each issue, provide:
- Issue type (spurious_feature, incorrect_answer, reasoning_failure)
- Severity (low, medium, high)
- Explanation of the issue
- Specific evidence from the response

Respond in JSON format with the following structure:
{
  "has_issues": true/false,
  "issues": [
    {
      "type": "spurious_feature" | "incorrect_answer" | "reasoning_failure",
      "severity": "low" | "medium" | "high",
      "description": "Description of the issue",
      "evidence": "Specific text from response showing the issue",
      "spurious_feature_name": "name of spurious feature if applicable"
    }
  ],
  "overall_assessment": "Brief summary"
}
""")
        
        # Add domain-specific instructions
        if domain == "medical":
            prompt_parts.append("""IMPORTANT: For medical questions, check if the model uses irrelevant demographic features (hair color, age stereotypes, gender assumptions, ethnicity) that should not influence medical diagnosis or advice.""")
        
        # Add spurious features to check
        if spurious_features:
            prompt_parts.append(f"\nSpecifically check for these spurious features: {', '.join(spurious_features)}")
        
        # Add question and response
        prompt_parts.append(f"\n### Question:\n{question}")
        
        if context:
            prompt_parts.append(f"\n### Context:\n{context}")
        
        prompt_parts.append(f"\n### Model Response:\n{model_response}")
        
        if ground_truth:
            prompt_parts.append(f"\n### Ground Truth Answer:\n{ground_truth}")
        
        prompt_parts.append("\n### Your Evaluation (JSON format):")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Format as Llama-3 chat
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{full_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
    
    def judge_response(
        self,
        question: str,
        model_response: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
        spurious_features: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Judge a model response.
        
        Returns:
            Dictionary with judgment results
        """
        prompt = self.create_judge_prompt(
            question=question,
            model_response=model_response,
            ground_truth=ground_truth,
            context=context,
            domain=domain,
            spurious_features=spurious_features,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        judgment_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Try to parse JSON from response
        judgment = self._parse_judgment(judgment_text)
        
        return {
            "judgment_text": judgment_text,
            "judgment": judgment,
        }
    
    def _parse_judgment(self, text: str) -> Dict[str, Any]:
        """Parse JSON judgment from text, with fallback."""
        import re
        
        # Try to extract JSON from the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: create a basic structure
        return {
            "has_issues": "issue" in text.lower() or "error" in text.lower() or "wrong" in text.lower(),
            "issues": [],
            "overall_assessment": text[:500],  # First 500 chars as assessment
        }
    
    def evaluate_responses(
        self,
        evaluation_results: List[Dict[str, Any]],
        spurious_features_map: Optional[Dict[str, List[str]]] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of model responses.
        
        Args:
            evaluation_results: Results from model evaluation
            spurious_features_map: Map from benchmark name to list of spurious features
            output_file: Optional file to save judgments
        
        Returns:
            List of judgments with failure flags
        """
        if spurious_features_map is None:
            spurious_features_map = {}
        
        judgments = []
        
        for result in tqdm(evaluation_results, desc="Judging responses"):
            benchmark = result.get("benchmark", "unknown")
            spurious_features = spurious_features_map.get(benchmark, None)
            
            judgment_result = self.judge_response(
                question=result["question"],
                model_response=result["model_response"],
                ground_truth=result.get("ground_truth"),
                context=result.get("context"),
                domain=result.get("domain", "general"),
                spurious_features=spurious_features,
            )
            
            judgment = judgment_result["judgment"]
            has_issues = judgment.get("has_issues", False)
            
            # Determine if this is a failure case
            is_failure = has_issues or len(judgment.get("issues", [])) > 0
            
            judgment_entry = {
                **result,
                "judgment": judgment,
                "judgment_text": judgment_result["judgment_text"],
                "is_failure": is_failure,
                "failure_type": self._categorize_failure(judgment),
            }
            
            judgments.append(judgment_entry)
            
            # Save incrementally
            if output_file and len(judgments) % 10 == 0:
                self._save_judgments(judgments, output_file)
        
        # Final save
        if output_file:
            self._save_judgments(judgments, output_file)
        
        return judgments
    
    def _categorize_failure(self, judgment: Dict[str, Any]) -> str:
        """Categorize the type of failure."""
        issues = judgment.get("issues", [])
        if not issues:
            return "none"
        
        issue_types = [issue.get("type", "") for issue in issues]
        
        if "spurious_feature" in issue_types:
            return "spurious_feature"
        elif "incorrect_answer" in issue_types:
            return "incorrect_answer"
        elif "reasoning_failure" in issue_types:
            return "reasoning_failure"
        else:
            return "other"
    
    def _save_judgments(self, judgments: List[Dict[str, Any]], output_file: str):
        """Save judgments to JSON file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)


def main(
    evaluation_file: str,
    output_file: str,
    judge_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    spurious_features_file: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Main judgment function.
    
    Args:
        evaluation_file: Path to evaluation results JSON
        output_file: Path to save judgments
        judge_model: Model to use as judge
        spurious_features_file: Optional JSON file mapping benchmarks to spurious features
        device: Device to use
    """
    # Load evaluation results
    print(f"Loading evaluation results from {evaluation_file}...")
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluation_results = json.load(f)
    
    print(f"Loaded {len(evaluation_results)} evaluation results")
    
    # Load spurious features map
    spurious_features_map = {}
    if spurious_features_file and os.path.exists(spurious_features_file):
        with open(spurious_features_file, 'r', encoding='utf-8') as f:
            spurious_features_map = json.load(f)
    
    # Initialize judge
    judge = JudgeEvaluator(judge_model=judge_model, device=device)
    
    # Evaluate
    judgments = judge.evaluate_responses(
        evaluation_results,
        spurious_features_map=spurious_features_map,
        output_file=output_file,
    )
    
    # Statistics
    failures = [j for j in judgments if j["is_failure"]]
    print(f"\nJudgment complete!")
    print(f"Total responses: {len(judgments)}")
    print(f"Failure cases: {len(failures)} ({100*len(failures)/len(judgments):.1f}%)")
    
    failure_types = {}
    for j in failures:
        ft = j["failure_type"]
        failure_types[ft] = failure_types.get(ft, 0) + 1
    
    print(f"\nFailure breakdown:")
    for ft, count in failure_types.items():
        print(f"  {ft}: {count}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
