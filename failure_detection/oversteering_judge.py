"""
Judge oversteering failures after spurious-feature steering.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OversteeringJudgeEvaluator:
    """Judge whether steering caused overcorrection failures."""

    def __init__(
        self,
        judge_model: str = "gpt-4.1",
        device: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.judge_model_name = judge_model
        self.temperature = temperature
        self.is_openai = self._is_openai_model(judge_model)

        if self.is_openai:
            if OpenAI is None:
                raise ImportError("openai package is required for OpenAI judge models. Install with: pip install openai")
            self.client = OpenAI()
            self.device = "api"
            print(f"Using OpenAI judge model {judge_model} via API.")
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    @staticmethod
    def _is_openai_model(model_name: str) -> bool:
        return model_name.startswith("gpt-4.1") or model_name.startswith("gpt-4")

    def _build_prompt_parts(
        self,
        question: str,
        model_response: str,
        spurious_feature: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
    ) -> Tuple[str, str]:
        feature = spurious_feature.replace("_", " ")
        instructions = f"""You are evaluating whether a spurious-feature steering intervention caused an overcorrection.

The model was steered to avoid relying on the spurious feature: "{feature}".
Your task: determine if the model's response is wrong *because it avoided or dismissed this feature when it was actually relevant*.

Respond in JSON format:
{{
  "is_oversteering_failure": true/false,
  "feature_relevant": true/false,
  "description": "Short explanation of the issue (if any)",
  "evidence": "Exact text from the response supporting the judgment",
  "suggested_correct_response": "Optional corrected response or key fact"
}}
"""
        prompt_parts = []

        if domain == "medical":
            prompt_parts.append(
                "For medical questions, ensure that medically relevant attributes are not ignored."
            )

        prompt_parts.append(f"\nSpurious feature under steering: {feature}")
        prompt_parts.append(f"\n### Question:\n{question}")
        if context:
            prompt_parts.append(f"\n### Context:\n{context}")
        prompt_parts.append(f"\n### Model Response:\n{model_response}")
        if ground_truth:
            prompt_parts.append(f"\n### Ground Truth Answer:\n{ground_truth}")
        prompt_parts.append("\n### Your Evaluation (JSON format):")

        body = "\n".join(prompt_parts)
        return instructions, body

    def _build_comparison_prompt_parts(
        self,
        question: str,
        baseline_response: str,
        steered_response: str,
        spurious_feature: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
    ) -> Tuple[str, str]:
        feature = spurious_feature.replace("_", " ")
        instructions = f"""You are comparing model outputs before and after a spurious-feature steering intervention.

The model was steered to avoid relying on the spurious feature: "{feature}".

Determine whether any new error in the *steered* response is attributable to this steering.

Respond in JSON format:
{{
  "is_oversteering_failure": true/false,
  "oversteering_due_to_steering": true/false,
  "feature_relevant": true/false,
  "description": "Short explanation of the issue (if any)",
  "evidence_before": "Exact text from the baseline response supporting your judgment",
  "evidence_after": "Exact text from the steered response supporting your judgment",
  "suggested_correct_response": "Optional corrected response or key fact"
}}
"""
        prompt_parts = []

        if domain == "medical":
            prompt_parts.append(
                "For medical questions, ensure that medically relevant attributes are not ignored."
            )

        prompt_parts.append(f"\nSpurious feature under steering: {feature}")
        prompt_parts.append(f"\n### Question:\n{question}")
        if context:
            prompt_parts.append(f"\n### Context:\n{context}")
        prompt_parts.append(f"\n### Baseline Response (before steering):\n{baseline_response}")
        prompt_parts.append(f"\n### Steered Response (after steering):\n{steered_response}")
        if ground_truth:
            prompt_parts.append(f"\n### Ground Truth Answer:\n{ground_truth}")
        prompt_parts.append("\n### Your Evaluation (JSON format):")

        body = "\n".join(prompt_parts)
        return instructions, body

    def _parse_judgment(self, text: str) -> Dict[str, Any]:
        import re
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {
            "is_oversteering_failure": False,
            "feature_relevant": False,
            "description": text[:500],
            "evidence": "",
            "suggested_correct_response": "",
        }

    def judge_response(
        self,
        question: str,
        model_response: str,
        spurious_feature: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        if self.is_openai:
            instructions, body = self._build_prompt_parts(
                question=question,
                model_response=model_response,
                spurious_feature=spurious_feature,
                ground_truth=ground_truth,
                context=context,
                domain=domain,
            )
            response = self.client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": body},
                ],
                temperature=self.temperature,
                max_tokens=max_new_tokens,
            )
            judgment_text = response.choices[0].message.content or ""
        else:
            instructions, body = self._build_prompt_parts(
                question=question,
                model_response=model_response,
                spurious_feature=spurious_feature,
                ground_truth=ground_truth,
                context=context,
                domain=domain,
            )
            prompt = f"{instructions}\n\n{body}"
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

        judgment = self._parse_judgment(judgment_text)
        return {
            "judgment_text": judgment_text,
            "judgment": judgment,
        }

    def judge_comparison(
        self,
        question: str,
        baseline_response: str,
        steered_response: str,
        spurious_feature: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        domain: str = "general",
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        if self.is_openai:
            instructions, body = self._build_comparison_prompt_parts(
                question=question,
                baseline_response=baseline_response,
                steered_response=steered_response,
                spurious_feature=spurious_feature,
                ground_truth=ground_truth,
                context=context,
                domain=domain,
            )
            response = self.client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": body},
                ],
                temperature=self.temperature,
                max_tokens=max_new_tokens,
            )
            judgment_text = response.choices[0].message.content or ""
        else:
            instructions, body = self._build_comparison_prompt_parts(
                question=question,
                baseline_response=baseline_response,
                steered_response=steered_response,
                spurious_feature=spurious_feature,
                ground_truth=ground_truth,
                context=context,
                domain=domain,
            )
            prompt = f"{instructions}\n\n{body}"
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

        judgment = self._parse_judgment(judgment_text)
        return {
            "judgment_text": judgment_text,
            "judgment": judgment,
        }

    def evaluate_responses(
        self,
        evaluation_results: List[Dict[str, Any]],
        spurious_feature: str,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        judgments = []

        for result in tqdm(evaluation_results, desc="Judging oversteering failures"):
            judgment_result = self.judge_response(
                question=result["question"],
                model_response=result["model_response"],
                ground_truth=result.get("ground_truth"),
                context=result.get("context"),
                domain=result.get("domain", "general"),
                spurious_feature=spurious_feature,
            )
            judgment = judgment_result["judgment"]
            is_failure = judgment.get("is_oversteering_failure", False)

            judgment_entry = {
                **result,
                "oversteering_judgment": judgment,
                "oversteering_judgment_text": judgment_result["judgment_text"],
                "is_oversteering_failure": is_failure,
            }
            judgments.append(judgment_entry)

            if output_file and len(judgments) % 10 == 0:
                self._save_judgments(judgments, output_file)

        if output_file:
            self._save_judgments(judgments, output_file)

        return judgments

    def evaluate_comparisons(
        self,
        baseline_results: List[Dict[str, Any]],
        steered_results: List[Dict[str, Any]],
        spurious_feature: str,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        baseline_by_id = {r["sample_id"]: r for r in baseline_results}
        judgments = []

        for result in tqdm(steered_results, desc="Judging baseline vs steered comparisons"):
            sample_id = result["sample_id"]
            baseline = baseline_by_id.get(sample_id)
            if not baseline:
                continue

            judgment_result = self.judge_comparison(
                question=result["question"],
                baseline_response=baseline["model_response"],
                steered_response=result["model_response"],
                ground_truth=result.get("ground_truth"),
                context=result.get("context"),
                domain=result.get("domain", "general"),
                spurious_feature=spurious_feature,
            )
            judgment = judgment_result["judgment"]
            is_failure = judgment.get("is_oversteering_failure", False)

            judgment_entry = {
                **result,
                "baseline_response": baseline["model_response"],
                "comparison_judgment": judgment,
                "comparison_judgment_text": judgment_result["judgment_text"],
                "is_oversteering_failure": is_failure,
            }
            judgments.append(judgment_entry)

            if output_file and len(judgments) % 10 == 0:
                self._save_judgments(judgments, output_file)

        if output_file:
            self._save_judgments(judgments, output_file)

        return judgments

    def _save_judgments(self, judgments: List[Dict[str, Any]], output_file: str):
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)


def main(
    evaluation_file: str,
    output_file: str,
    spurious_feature: str = "hair color",
    judge_model: str = "gpt-4.1",
    device: Optional[str] = None,
):
    print(f"Loading evaluation results from {evaluation_file}...")
    with open(evaluation_file, "r", encoding="utf-8") as f:
        evaluation_results = json.load(f)
    print(f"Loaded {len(evaluation_results)} evaluation results")

    judge = OversteeringJudgeEvaluator(judge_model=judge_model, device=device)
    judgments = judge.evaluate_responses(
        evaluation_results,
        spurious_feature=spurious_feature,
        output_file=output_file,
    )

    failures = [j for j in judgments if j.get("is_oversteering_failure")]
    print(f"\nJudgment complete!")
    print(f"Total responses: {len(judgments)}")
    print(f"Oversteering failures: {len(failures)} ({100*len(failures)/len(judgments):.1f}%)")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
