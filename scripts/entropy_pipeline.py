"""
Entropy-Based Step Verification Pipeline
=========================================

This is the CORE of the project. Here's what it does:

1. Feeds a math problem to an SLM
2. The model generates a step-by-step solution
3. At EACH TOKEN the model generates, we capture its probability distribution
4. We compute entropy from that distribution (how uncertain was the model?)
5. We group tokens into reasoning steps
6. We compute average entropy PER STEP
7. High entropy steps = model was uncertain = these need verification
   Low entropy steps = model was confident = safe to skip

The entropy formula:
    H(X) = -sum( p(x) * log2(p(x)) )

Where p(x) is the probability of each possible next token.

If the model is 100% sure of the next token: H = 0 (minimum entropy)
If the model is equally unsure between all tokens: H = log2(vocab_size) (maximum entropy)

In practice, most steps will have entropy somewhere in between.
"""

import json
import os
import re
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PART 1: ENTROPY COMPUTATION
# ============================================================
# This is the mathematical heart of the project.

def compute_token_entropy(logits):
    """
    Compute entropy from model logits for a single token position.
    
    Args:
        logits: Raw model output scores, shape (vocab_size,)
                These are NOT probabilities yet - they're raw scores.
    
    Returns:
        float: Entropy in bits (using log base 2)
    
    How it works:
    1. Convert logits to probabilities using softmax
       softmax(x_i) = exp(x_i) / sum(exp(x_j))
       This ensures all values are 0-1 and sum to 1.
    
    2. Compute entropy: H = -sum(p * log2(p))
       - If one probability is ~1.0 and rest are ~0: H is near 0 (confident)
       - If probabilities are spread out: H is high (uncertain)
    """
    # Step 1: Softmax to convert logits to probabilities
    probs = torch.softmax(logits.float(), dim=-1)
    
    # Step 2: Compute entropy
    # We add a tiny epsilon to avoid log(0) which is undefined
    log_probs = torch.log2(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs).item()
    
    return entropy


def compute_step_entropy(token_entropies, step_token_indices):
    """
    Compute the average entropy for a reasoning step.
    
    A step might be: "She eats 3, leaving 16 - 3 = 13"
    This step is made up of ~15 tokens.
    Each token has its own entropy.
    We average them to get the step's overall uncertainty.
    
    Args:
        token_entropies: List of entropy values, one per generated token
        step_token_indices: (start, end) indices for this step's tokens
    
    Returns:
        dict with mean, max, and min entropy for the step
    """
    start, end = step_token_indices
    step_entropies = token_entropies[start:end]
    
    if not step_entropies:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "num_tokens": 0}
    
    return {
        "mean": float(np.mean(step_entropies)),
        "max": float(np.max(step_entropies)),
        "min": float(np.min(step_entropies)),
        "num_tokens": len(step_entropies)
    }


# ============================================================
# PART 2: GENERATION WITH ENTROPY CAPTURE
# ============================================================
# Normal generation just gives you text.
# We need to generate token-by-token and capture logits at each step.

def generate_with_entropy(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generate text token by token, capturing entropy at each position.
    
    Normal model.generate() is a black box - text goes in, text comes out.
    Here we do it manually:
    1. Encode the prompt
    2. For each new token:
       a. Run the model to get logits (probability distribution)
       b. Compute entropy from those logits
       c. Pick the most likely token (greedy) or sample
       d. Append it and repeat
    
    This is slower than model.generate() but gives us the entropy data
    we need for the entire project.
    
    Args:
        model: The loaded language model
        tokenizer: The model's tokenizer
        prompt: The formatted prompt string
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        dict with:
            - generated_text: The full generated text
            - token_entropies: List of entropy values, one per token
            - tokens: List of actual token strings generated
            - total_entropy: Sum of all token entropies
            - mean_entropy: Average entropy across all tokens
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Storage for our entropy data
    token_entropies = []
    generated_tokens = []
    generated_ids = []
    
    # Generate token by token
    current_ids = input_ids.clone()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Forward pass: get the model's prediction
            outputs = model(current_ids)
            
            # logits shape: (batch_size, sequence_length, vocab_size)
            # We only care about the LAST position (the prediction for next token)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Compute entropy for this token position
            entropy = compute_token_entropy(next_token_logits)
            token_entropies.append(entropy)
            
            # Greedy decoding: pick the most probable token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Decode the token to text
            token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=False)
            generated_tokens.append(token_text)
            generated_ids.append(next_token_id.item())
            
            # Check if we hit the end-of-sequence token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Append the new token and continue
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)
    
    # Decode the full generated text
    full_output_ids = current_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(full_output_ids, skip_special_tokens=True)
    
    return {
        "generated_text": generated_text,
        "token_entropies": token_entropies,
        "tokens": generated_tokens,
        "total_entropy": float(sum(token_entropies)),
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0.0
    }


# ============================================================
# PART 3: STEP PARSING AND ENTROPY MAPPING
# ============================================================
# The model generates one long string. We need to split it into
# reasoning steps and map entropy values to each step.

def parse_steps_from_output(generated_text, tokens):
    """
    Split generated text into reasoning steps and find which tokens
    belong to which step.
    
    Models typically separate steps with newlines, numbers, or markers like
    "Step 1:", "1.", "First,", etc.
    
    Args:
        generated_text: The full model output text
        tokens: List of individual token strings
    
    Returns:
        List of dicts, each containing:
            - step_num: Which step this is (1, 2, 3...)
            - text: The text of this step
            - token_start: Index of first token in this step
            - token_end: Index of last token in this step
    """
    # Split by common step delimiters
    # Models often use numbered steps or newlines
    step_patterns = [
        r'\n\s*\d+[\.\)]\s*',      # "1." or "1)" at start of line
        r'\n\s*Step\s*\d+[:\.]',     # "Step 1:" or "Step 1."
        r'\n\s*\*\*Step\s*\d+',      # "**Step 1"
        r'\n\s*First[,:]',           # "First,"
        r'\n\s*Second[,:]',          # "Second,"
        r'\n\s*Then[,:]',            # "Then,"
        r'\n\s*Finally[,:]',         # "Finally,"
    ]
    
    # Try splitting by numbered steps first
    steps_text = re.split(r'\n\s*(?:\d+[\.\)]|\*\*?\s*(?:Step\s*)?\d+)', generated_text)
    
    # Filter empty steps
    steps_text = [s.strip() for s in steps_text if s.strip()]
    
    if len(steps_text) <= 1:
        # If no numbered steps found, split by double newlines
        steps_text = [s.strip() for s in generated_text.split('\n\n') if s.strip()]
    
    if len(steps_text) <= 1:
        # Last resort: split by single newlines
        steps_text = [s.strip() for s in generated_text.split('\n') if s.strip()]
    
    # Now map tokens to steps
    # We reconstruct the text token by token to find boundaries
    steps = []
    current_token_idx = 0
    reconstructed = ""
    
    for step_num, step_text in enumerate(steps_text):
        step_start = current_token_idx
        
        # Find how many tokens make up this step
        # by accumulating tokens until we've covered this step's text
        step_target = step_text[:50]  # Match first 50 chars of the step
        
        tokens_in_step = 0
        step_reconstructed = ""
        
        while current_token_idx < len(tokens) and tokens_in_step < 200:
            step_reconstructed += tokens[current_token_idx]
            current_token_idx += 1
            tokens_in_step += 1
            
            # Check if we've covered enough of this step
            if step_target and step_target[:20] in step_reconstructed:
                # Continue until next step boundary or reasonable length
                # Look ahead for step boundary markers
                remaining_text = ""
                lookahead = current_token_idx
                for j in range(min(10, len(tokens) - current_token_idx)):
                    remaining_text += tokens[current_token_idx + j] if current_token_idx + j < len(tokens) else ""
                
                if any(marker in remaining_text for marker in ['\n1', '\n2', '\n3', '\nStep', '\n**']):
                    break
        
        steps.append({
            "step_num": step_num + 1,
            "text": step_text[:200],  # Truncate for readability
            "token_start": step_start,
            "token_end": current_token_idx
        })
    
    return steps


# ============================================================
# PART 4: FULL PIPELINE
# ============================================================

def run_entropy_pipeline(model_name, model, tokenizer, problems, output_dir="results"):
    """
    Run the full entropy measurement pipeline on a set of problems.
    
    For each problem:
    1. Format the prompt
    2. Generate with entropy capture
    3. Parse into steps
    4. Compute per-step entropy
    5. Save everything
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for i, problem in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(problems)}: {problem['question'][:80]}...")
        
        # Format prompt
        prompt = f"Solve this step by step:\n{problem['question']}\nShow your reasoning at each step."
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate with entropy capture
        start_time = time.time()
        gen_result = generate_with_entropy(model, tokenizer, formatted_prompt, max_new_tokens=512)
        inference_time = time.time() - start_time
        
        print(f"  Generated {len(gen_result['tokens'])} tokens in {inference_time:.1f}s")
        print(f"  Mean entropy: {gen_result['mean_entropy']:.3f} bits")
        
        # Parse into steps
        steps = parse_steps_from_output(gen_result['generated_text'], gen_result['tokens'])
        
        # Compute per-step entropy
        step_data = []
        for step in steps:
            step_entropy = compute_step_entropy(
                gen_result['token_entropies'],
                (step['token_start'], step['token_end'])
            )
            
            step_info = {
                "step_num": step['step_num'],
                "text": step['text'],
                "entropy": step_entropy,
                "token_range": [step['token_start'], step['token_end']]
            }
            step_data.append(step_info)
            
            print(f"  Step {step['step_num']}: entropy={step_entropy['mean']:.3f} "
                  f"(max={step_entropy['max']:.3f}, tokens={step_entropy['num_tokens']})")
        
        # Try to extract the model's final answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', gen_result['generated_text'])
        if boxed_match:
            model_answer = boxed_match.group(1).replace(",", "").strip()
        else:
            numbers = re.findall(r'[\d,]+\.?\d*', gen_result['generated_text'])
            model_answer = numbers[-1].replace(",", "") if numbers else "unknown"
        
        expected = problem.get('final_answer', '').replace(",", "").strip()
        is_correct = model_answer == expected
        
        print(f"  Answer: {model_answer} (expected: {expected}) {'CORRECT' if is_correct else 'WRONG'}")
        
        result = {
            "id": problem.get("id", f"problem_{i}"),
            "question": problem["question"],
            "expected_answer": expected,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "generated_text": gen_result['generated_text'],
            "mean_entropy": gen_result['mean_entropy'],
            "total_entropy": gen_result['total_entropy'],
            "num_tokens_generated": len(gen_result['tokens']),
            "steps": step_data,
            "num_steps": len(step_data),
            "token_entropies": gen_result['token_entropies']  # Raw per-token data
        }
        
        all_results.append(result)
    
    # Save results
    safe_name = model_name.replace("/", "_").replace("-", "_")
    output_path = os.path.join(output_dir, f"{safe_name}_entropy.json")
    
    # Save without the raw token_entropies for the summary file (too large)
    summary_results = []
    for r in all_results:
        summary = {k: v for k, v in r.items() if k != 'token_entropies'}
        summary_results.append(summary)
    
    with open(output_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    
    # Also save the full data with token entropies
    full_output_path = os.path.join(output_dir, f"{safe_name}_entropy_full.json")
    with open(full_output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ENTROPY ANALYSIS SUMMARY: {model_name}")
    print(f"{'='*60}")
    correct = sum(1 for r in all_results if r['is_correct'])
    total = len(all_results)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Avg entropy per problem: {np.mean([r['mean_entropy'] for r in all_results]):.3f}")
    print(f"Avg steps per problem: {np.mean([r['num_steps'] for r in all_results]):.1f}")
    
    # Show entropy distribution across all steps
    all_step_entropies = []
    for r in all_results:
        for s in r['steps']:
            all_step_entropies.append(s['entropy']['mean'])
    
    if all_step_entropies:
        print(f"\nStep entropy distribution:")
        print(f"  Mean: {np.mean(all_step_entropies):.3f}")
        print(f"  Std:  {np.std(all_step_entropies):.3f}")
        print(f"  Min:  {np.min(all_step_entropies):.3f}")
        print(f"  Max:  {np.max(all_step_entropies):.3f}")
        
        # Classify steps by entropy
        threshold = np.mean(all_step_entropies)
        high_entropy = sum(1 for e in all_step_entropies if e > threshold)
        low_entropy = sum(1 for e in all_step_entropies if e <= threshold)
        print(f"\n  High entropy steps (above mean): {high_entropy}")
        print(f"  Low entropy steps (below mean):  {low_entropy}")
        print(f"  Ratio: {high_entropy/(high_entropy+low_entropy)*100:.0f}% high / {low_entropy/(high_entropy+low_entropy)*100:.0f}% low")
    
    print(f"\nResults saved to:")
    print(f"  Summary: {output_path}")
    print(f"  Full data: {full_output_path}")
    
    return all_results


# ============================================================
# PART 5: MAIN - RUN IT
# ============================================================

MODELS = {
    "qwen-math-1.5b": {
        "name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "size": "1.5B",
        "family": "Alibaba"
    },
    "llama-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "size": "3B",
        "family": "Meta"
    },
    "gemma3-4b": {
        "name": "google/gemma-3-4b-it",
        "size": "4B",
        "family": "Google"
    }
}


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python entropy_pipeline.py <model_key> [num_problems]")
        print(f"Available models: {', '.join(MODELS.keys())}")
        print("\nExample:")
        print("  python scripts/entropy_pipeline.py qwen-math-1.5b 10")
        return
    
    model_key = sys.argv[1]
    num_problems = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return
    
    info = MODELS[model_key]
    
    # Load model
    print(f"Loading {info['family']} {model_key} ({info['size']})...")
    tokenizer = AutoTokenizer.from_pretrained(info["name"], trust_remote_code=True)
    
    # Use bfloat16 for better numerical stability (especially for Gemma)
    model = AutoModelForCausalLM.from_pretrained(
        info["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!")
    
    # Load GSM8K problems
    with open("data/gsm8k/test.json") as f:
        all_problems = json.load(f)
    
    problems = all_problems[:num_problems]
    print(f"Running {len(problems)} problems...")
    
    # Run the pipeline
    results = run_entropy_pipeline(model_key, model, tokenizer, problems)
    
    print(f"\nDone! Push to GitHub:")
    print(f"  git add . && git commit -m 'add {model_key} entropy results' && git push")


if __name__ == "__main__":
    main()