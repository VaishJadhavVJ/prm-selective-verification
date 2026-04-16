"""
Real Verification Pipeline
===========================

Two real verifiers replacing the simulated one:

1. Math-Shepherd PRM (Qwen2-0.5B fine-tuned)
   - Trained specifically on math step verification
   - Scores each step as correct/incorrect
   - Only works for math (GSM8K)

2. Gemini LLM-as-Judge
   - Uses Google Gemini API to evaluate step correctness
   - Works on ANY domain (math, finance, anything)
   - Slower (API calls) but more flexible

We run BOTH on GSM8K to compare them (per professor's request).
We run only Gemini on options trading.
"""

import json
import os
import time
import re
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


# ============================================================
# VERIFIER 1: MATH-SHEPHERD PRM
# ============================================================
# This is a real PRM. It was trained on the Math-Shepherd dataset
# to classify each reasoning step as correct (+) or incorrect (-).
#
# How it works:
# - Input: question + step-by-step solution with step markers
# - Output: probability score per step (0 to 1)
# - Score > 0.5 = step is likely correct
# - Score < 0.5 = step is likely wrong

class MathShepherdPRM:
    def __init__(self, model_name="trl-lib/Qwen2-0.5B-Reward-Math-Sheperd"):
        """Load the PRM model."""
        print(f"Loading Math-Shepherd PRM: {model_name}...")
        self.pipe = pipeline(
            "token-classification",
            model=model_name,
            device=-1  # CPU (safer for token-classification)
        )
        print("PRM loaded!")
    
    def score_steps(self, question, steps_text):
        """
        Score each reasoning step.
        
        Args:
            question: The math problem text
            steps_text: List of step strings
        
        Returns:
            List of dicts with step scores
        """
        scores = []
        
        for idx in range(1, len(steps_text) + 1):
            # Build input: question + steps up to current step
            text = "\n".join([question] + steps_text[:idx]) + "\n"
            
            try:
                output = self.pipe(text)
                if output:
                    # Last token classification gives the step score
                    score = float(output[-1]["score"])
                    label = output[-1]["entity"]
                    is_correct = label == "LABEL_1"
                else:
                    score = 0.5
                    is_correct = True
            except Exception as e:
                print(f"    PRM error on step {idx}: {e}")
                score = 0.5
                is_correct = True
            
            scores.append({
                "step_num": idx,
                "score": score,
                "is_correct": is_correct,
                "verifier": "math_shepherd_prm"
            })
        
        return scores


# ============================================================
# VERIFIER 2: GEMINI LLM-AS-JUDGE
# ============================================================
# Uses Google Gemini API to evaluate each step.
# We prompt it to act as a math/reasoning verifier.
#
# This is the approach used when no domain-specific PRM exists.
# It works on any domain: math, finance, science, etc.

class GeminiJudge:
    def __init__(self, api_key):
        """Initialize Gemini API."""
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"
        print("Gemini Judge initialized!")
    
    def score_steps(self, question, steps_text):
        """
        Score each reasoning step using Gemini.
        
        For each step, we ask Gemini:
        "Given this problem and previous steps, is this step correct?"
        
        Returns a score from 0 to 1.
        """
        scores = []
        
        for idx, step in enumerate(steps_text):
            # Build context
            previous_steps = "\n".join(steps_text[:idx]) if idx > 0 else "(First step)"
            
            prompt = f"""You are a precise mathematical reasoning verifier. 

Problem: {question}

Previous steps:
{previous_steps}

Current step to verify (Step {idx + 1}):
{step}

Is this step mathematically correct? Consider:
1. Are the calculations accurate?
2. Does it follow logically from the previous steps?
3. Is the reasoning sound?

Respond with ONLY a JSON object in this exact format:
{{"score": <number between 0.0 and 1.0>, "correct": <true or false>, "reason": "<brief explanation>"}}

Where score 1.0 means definitely correct and 0.0 means definitely wrong."""

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                response_text = response.text.strip()
                
                # Parse JSON from response
                # Clean up markdown code blocks if present
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                
                result = json.loads(response_text)
                score = float(result.get("score", 0.5))
                is_correct = result.get("correct", True)
                reason = result.get("reason", "")
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract score from text
                try:
                    score_match = re.search(r'"score"\s*:\s*([\d.]+)', response_text)
                    if score_match:
                        score = float(score_match.group(1))
                        is_correct = score > 0.5
                        reason = "Parsed from malformed JSON"
                    else:
                        score = 0.5
                        is_correct = True
                        reason = "Could not parse response"
                except:
                    score = 0.5
                    is_correct = True
                    reason = "Error parsing"
                    
            except Exception as e:
                print(f"    Gemini error on step {idx + 1}: {e}")
                score = 0.5
                is_correct = True
                reason = f"API error: {str(e)}"
                # Rate limiting - wait a bit
                time.sleep(2)
            
            scores.append({
                "step_num": idx + 1,
                "score": score,
                "is_correct": is_correct,
                "reason": reason if 'reason' in dir() else "",
                "verifier": "gemini_judge"
            })
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return scores


# ============================================================
# VERIFICATION WITH STRATEGIES
# ============================================================

def apply_verification_strategies(problem_result, verifier, verifier_name):
    """
    Apply all three verification strategies using a real verifier.
    
    Args:
        problem_result: Output from entropy_pipeline (has steps with entropy)
        verifier: Either MathShepherdPRM or GeminiJudge instance
        verifier_name: String name for logging
    
    Returns:
        Dict with results from all three strategies
    """
    # Get non-empty steps
    steps = [s for s in problem_result['steps'] if s['entropy']['num_tokens'] > 0]
    
    if not steps:
        return None
    
    # Extract step text for the verifier
    steps_text = [s['text'] for s in steps]
    
    # ---- STRATEGY 1: FULL VERIFICATION ----
    # Score ALL steps
    all_scores = verifier.score_steps(problem_result['question'], steps_text)
    
    full_result = {
        'strategy': 'full',
        'verifier': verifier_name,
        'scores': all_scores,
        'num_verified': len(all_scores),
        'total_steps': len(steps),
        'verification_ratio': 1.0,
        'min_score': min(s['score'] for s in all_scores),
        'mean_score': np.mean([s['score'] for s in all_scores]),
        'flagged_steps': [s['step_num'] for s in all_scores if s['score'] < 0.5]
    }
    
    # ---- STRATEGY 2: RANDOM VERIFICATION ----
    import random
    random.seed(42)
    num_to_verify = max(1, len(steps) // 2)
    random_indices = sorted(random.sample(range(len(steps)), num_to_verify))
    
    random_scores = []
    for i, step in enumerate(steps):
        if i in random_indices:
            # Use the score we already computed (avoid duplicate API calls)
            random_scores.append(all_scores[i])
        else:
            random_scores.append({
                'step_num': i + 1,
                'score': None,
                'is_correct': None,
                'verifier': verifier_name,
                'skipped': True
            })
    
    verified_random = [s for s in random_scores if s.get('score') is not None]
    
    random_result = {
        'strategy': 'random',
        'verifier': verifier_name,
        'scores': random_scores,
        'num_verified': len(verified_random),
        'total_steps': len(steps),
        'verification_ratio': len(verified_random) / len(steps),
        'min_score': min(s['score'] for s in verified_random) if verified_random else 0,
        'mean_score': np.mean([s['score'] for s in verified_random]) if verified_random else 0,
        'flagged_steps': [s['step_num'] for s in verified_random if s['score'] < 0.5]
    }
    
    # ---- STRATEGY 3: ENTROPY-BASED SELECTIVE ----
    entropies = [s['entropy']['mean'] for s in steps]
    threshold = np.mean(entropies)
    
    entropy_scores = []
    for i, step in enumerate(steps):
        if step['entropy']['mean'] > threshold:
            # High entropy - verify it
            entropy_scores.append(all_scores[i])
        else:
            # Low entropy - skip
            entropy_scores.append({
                'step_num': i + 1,
                'score': None,
                'is_correct': None,
                'verifier': verifier_name,
                'skipped': True,
                'entropy': step['entropy']['mean'],
                'reason': f"Entropy {step['entropy']['mean']:.3f} below threshold {threshold:.3f}"
            })
    
    verified_entropy = [s for s in entropy_scores if s.get('score') is not None]
    
    entropy_result = {
        'strategy': 'entropy_based',
        'verifier': verifier_name,
        'entropy_threshold': float(threshold),
        'scores': entropy_scores,
        'num_verified': len(verified_entropy),
        'total_steps': len(steps),
        'verification_ratio': len(verified_entropy) / len(steps) if steps else 0,
        'min_score': min(s['score'] for s in verified_entropy) if verified_entropy else 1.0,
        'mean_score': np.mean([s['score'] for s in verified_entropy]) if verified_entropy else 1.0,
        'flagged_steps': [s['step_num'] for s in verified_entropy if s['score'] < 0.5]
    }
    
    return {
        'problem_id': problem_result.get('id', 'unknown'),
        'question': problem_result['question'][:100],
        'is_correct': problem_result['is_correct'],
        'full': full_result,
        'random': random_result,
        'entropy_based': entropy_result
    }


# ============================================================
# MAIN
# ============================================================

def main():
    import sys
    
    # Get Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
    
    # Available entropy result files
    model_files = {
        'qwen-math-1.5b': 'results/qwen_math_1.5b_entropy.json',
        'llama-3b': 'results/llama_3b_entropy.json',
        'gemma3-4b': 'results/gemma3_4b_entropy.json'
    }
    
    # Pick model to verify
    model_key = sys.argv[1] if len(sys.argv) > 1 else 'qwen-math-1.5b'
    
    if model_key not in model_files:
        print(f"Available models: {', '.join(model_files.keys())}")
        return
    
    results_path = model_files[model_key]
    if not os.path.exists(results_path):
        print(f"No entropy results at {results_path}. Run entropy_pipeline.py first.")
        return
    
    with open(results_path) as f:
        entropy_results = json.load(f)
    
    print(f"Loaded {len(entropy_results)} problems from {results_path}")
    
    # ---- Initialize verifiers ----
    
    # Verifier 1: Math-Shepherd PRM
    print("\n--- Setting up Math-Shepherd PRM ---")
    try:
        prm = MathShepherdPRM()
        has_prm = True
    except Exception as e:
        print(f"Could not load PRM: {e}")
        print("Will use Gemini only.")
        has_prm = False
    
    # Verifier 2: Gemini Judge
    print("\n--- Setting up Gemini Judge ---")
    gemini = GeminiJudge(api_key)
    
    # ---- Run verification on each problem ----
    all_results = []
    
    for i, problem in enumerate(entropy_results):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(entropy_results)}: {problem['question'][:80]}...")
        print(f"Model answer: {problem['model_answer']} (Expected: {problem['expected_answer']}) "
              f"{'CORRECT' if problem['is_correct'] else 'WRONG'}")
        
        result = {'problem_id': problem.get('id', f'problem_{i}')}
        
        # Run PRM verification
        if has_prm:
            print(f"\n  Running Math-Shepherd PRM...")
            prm_result = apply_verification_strategies(problem, prm, "math_shepherd_prm")
            if prm_result:
                result['prm'] = prm_result
                print(f"    Full: {prm_result['full']['num_verified']} steps, "
                      f"min_score={prm_result['full']['min_score']:.3f}")
                print(f"    Entropy: {prm_result['entropy_based']['num_verified']} steps, "
                      f"min_score={prm_result['entropy_based']['min_score']:.3f}")
        
        # Run Gemini verification
        print(f"\n  Running Gemini Judge...")
        gemini_result = apply_verification_strategies(problem, gemini, "gemini_judge")
        if gemini_result:
            result['gemini'] = gemini_result
            print(f"    Full: {gemini_result['full']['num_verified']} steps, "
                  f"min_score={gemini_result['full']['min_score']:.3f}")
            print(f"    Entropy: {gemini_result['entropy_based']['num_verified']} steps, "
                  f"min_score={gemini_result['entropy_based']['min_score']:.3f}")
        
        all_results.append(result)
    
    # ---- Save results ----
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{model_key}_verification.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # ---- Print summary ----
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY: {model_key}")
    print(f"{'='*70}")
    
    for verifier_name in ['prm', 'gemini']:
        results_with_verifier = [r for r in all_results if verifier_name in r]
        if not results_with_verifier:
            continue
        
        label = "Math-Shepherd PRM" if verifier_name == 'prm' else "Gemini Judge"
        print(f"\n{label}:")
        print(f"  {'Strategy':<20} {'Steps Verified':<18} {'Ratio':<10} {'Flagged'}")
        print(f"  {'-'*60}")
        
        for strat in ['full', 'random', 'entropy_based']:
            verified = []
            ratios = []
            flagged = []
            
            for r in results_with_verifier:
                s = r[verifier_name][strat]
                verified.append(s['num_verified'])
                ratios.append(s['verification_ratio'])
                flagged.append(len(s['flagged_steps']))
            
            avg_v = np.mean(verified)
            avg_r = np.mean(ratios)
            avg_f = np.mean(flagged)
            print(f"  {strat:<20} {avg_v:<18.1f} {avg_r:<10.1%} {avg_f:.1f}")
    
    # ---- Compare PRM vs Gemini ----
    if has_prm:
        print(f"\n{'='*70}")
        print(f"PRM vs GEMINI AGREEMENT (Full Verification)")
        print(f"{'='*70}")
        
        for r in all_results:
            if 'prm' in r and 'gemini' in r:
                prm_scores = [s['score'] for s in r['prm']['full']['scores']]
                gem_scores = [s['score'] for s in r['gemini']['full']['scores']]
                
                # Compare flagged steps
                prm_flags = set(r['prm']['full']['flagged_steps'])
                gem_flags = set(r['gemini']['full']['flagged_steps'])
                agreement = prm_flags == gem_flags
                
                print(f"  {r['problem_id']}: PRM flagged {prm_flags or 'none'}, "
                      f"Gemini flagged {gem_flags or 'none'} "
                      f"{'AGREE' if agreement else 'DISAGREE'}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nPush: git add . && git commit -m 'add real verification results for {model_key}' && git push")


if __name__ == "__main__":
    main()