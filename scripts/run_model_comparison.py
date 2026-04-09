"""
Run multiple models on GSM8K problems for comparison.
Downloads each model on first run.
Saves outputs to results/
"""

import json
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "size": "3B",
        "family": "Meta"
    },
    "gemma3-4b": {
        "name": "google/gemma-3-4b-it",
        "size": "4B",
        "family": "Google"
    },
    "qwen-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "size": "3B",
        "family": "Alibaba"
    }
}


def load_model(model_key):
    """Load a model and tokenizer."""
    info = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {info['family']} {model_key} ({info['size']})...")
    print(f"Model: {info['name']}")
    print(f"First run will download the model weights.")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        info["name"],
        trust_remote_code=True
    )

    # Use float16 for efficiency on Apple Silicon
    model = AutoModelForCausalLM.from_pretrained(
    info["name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

    print(f"{model_key} loaded successfully!")
    return model, tokenizer


def run_problems(model, tokenizer, problems, model_key):
    """Run model on GSM8K problems and collect outputs."""
    info = MODELS[model_key]
    results = []

    for i, problem in enumerate(problems):
        print(f"\n  Problem {i+1}/{len(problems)}: {problem['question'][:80]}...")

        prompt = f"Solve this step by step:\n{problem['question']}\nShow your reasoning at each step."

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start_time = time.time()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        inference_time = time.time() - start_time

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Try to extract final numerical answer from model output
        # Look for \boxed{} pattern or last number
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            model_answer = boxed_match.group(1).replace(",", "").strip()
        else:
            # Try to find last number in response
            numbers = re.findall(r'[\d,]+\.?\d*', response)
            model_answer = numbers[-1].replace(",", "") if numbers else "unknown"

        expected = problem["final_answer"].replace(",", "").strip()
        is_correct = model_answer == expected

        print(f"  Expected: {expected} | Model: {model_answer} | {'✓' if is_correct else '✗'} | Time: {inference_time:.1f}s")

        results.append({
            "id": problem["id"],
            "question": problem["question"],
            "expected_answer": expected,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "model_output": response,
            "model": model_key,
            "model_family": info["family"],
            "model_size": info["size"]
        })

    return results


def print_summary(all_results):
    """Print comparison summary across models."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    for model_key in all_results:
        results = all_results[model_key]
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        avg_time = sum(r["inference_time"] for r in results) / total
        info = MODELS[model_key]

        print(f"\n{info['family']} {model_key} ({info['size']}):")
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
        print(f"  Avg inference time: {avg_time:.1f}s per problem")


def main():
    import sys

    # Check which model to run
    if len(sys.argv) > 1:
        model_keys = [sys.argv[1]]
        if model_keys[0] not in MODELS:
            print(f"Unknown model: {model_keys[0]}")
            print(f"Available: {', '.join(MODELS.keys())}")
            return
    else:
        print("Usage: python run_model_comparison.py <model_key>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        print("\nRun one at a time to avoid memory issues:")
        print("  python scripts/run_model_comparison.py phi4-mini")
        print("  python scripts/run_model_comparison.py gemma3-12b")
        return

    # Load problems
    with open("data/gsm8k/test.json") as f:
        all_problems = json.load(f)

    problems = all_problems[:10]
    print(f"Running {len(problems)} GSM8K problems")

    # Run selected model
    model_key = model_keys[0]
    model, tokenizer = load_model(model_key)
    results = run_problems(model, tokenizer, problems, model_key)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{model_key}_gsm8k_outputs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    avg_time = sum(r["inference_time"] for r in results) / total

    print(f"\n{'='*60}")
    print(f"RESULTS: {MODELS[model_key]['family']} {model_key} ({MODELS[model_key]['size']})")
    print(f"  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Avg inference time: {avg_time:.1f}s per problem")
    print(f"  Saved to {output_path}")
    print(f"\nPush to GitHub: git add . && git commit -m 'add {model_key} results' && git push")


if __name__ == "__main__":
    main()
