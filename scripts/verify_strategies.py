"""
Three Verification Strategies Comparison
=========================================

This implements the three algorithms we compare:

1. FULL VERIFICATION: Check every single step (baseline)
2. RANDOM VERIFICATION: Check a random subset of steps  
3. ENTROPY-BASED SELECTIVE: Check only high-entropy steps (our method)

Each strategy has a "verification budget" - how many steps do we verify.
We measure:
- Accuracy: Does it catch the same errors as full verification?
- Compute cost: How many verification calls did we make?

For now, since we don't have the PRM yet, we use a SIMULATED verifier:
- It checks if the step contains the correct intermediate answer
- This gives us a ground truth to compare strategies against
"""

import json
import os
import random
import numpy as np
from collections import defaultdict


# ============================================================
# SIMULATED VERIFIER
# ============================================================
# In the real project, we'll use Math-Shepherd PRM.
# For now, we simulate by checking if the step's arithmetic
# matches what's expected.

def simulated_verify_step(step, is_problem_correct):
    """
    Simulated step verifier.
    
    In reality, a PRM would score each step on a scale (e.g., 0-1).
    For this simulation:
    - If the problem's final answer was correct, assume most steps are correct
    - If wrong, assume at least one step was wrong
    - Add some noise to make it realistic
    
    Returns: score between 0 and 1 (1 = correct, 0 = wrong)
    """
    if is_problem_correct:
        # If problem is correct, steps likely correct but with some noise
        return np.random.uniform(0.7, 1.0)
    else:
        # If problem is wrong, at least one step is likely bad
        return np.random.uniform(0.3, 0.8)


# ============================================================
# STRATEGY 1: FULL VERIFICATION
# ============================================================

def full_verification(problem_result):
    """
    Verify every single step. Maximum cost, maximum information.
    This is the baseline everyone currently uses.
    """
    steps = problem_result['steps']
    verifications = []
    
    for step in steps:
        # Skip empty steps (artifacts of parsing)
        if step['entropy']['num_tokens'] == 0:
            continue
            
        score = simulated_verify_step(step, problem_result['is_correct'])
        verifications.append({
            'step_num': step['step_num'],
            'entropy': step['entropy']['mean'],
            'verified': True,
            'score': score
        })
    
    return {
        'strategy': 'full',
        'verifications': verifications,
        'num_verified': len(verifications),
        'total_steps': len([s for s in steps if s['entropy']['num_tokens'] > 0]),
        'verification_ratio': 1.0,
        'final_decision_correct': all(v['score'] > 0.5 for v in verifications) == problem_result['is_correct']
    }


# ============================================================
# STRATEGY 2: RANDOM VERIFICATION
# ============================================================

def random_verification(problem_result, budget_ratio=0.5):
    """
    Verify a random subset of steps.
    This is a control to test if simply reducing verification helps.
    
    budget_ratio: fraction of steps to verify (0.5 = half)
    """
    steps = [s for s in problem_result['steps'] if s['entropy']['num_tokens'] > 0]
    
    if not steps:
        return None
    
    num_to_verify = max(1, int(len(steps) * budget_ratio))
    selected = random.sample(steps, num_to_verify)
    selected_nums = {s['step_num'] for s in selected}
    
    verifications = []
    for step in steps:
        if step['step_num'] in selected_nums:
            score = simulated_verify_step(step, problem_result['is_correct'])
            verifications.append({
                'step_num': step['step_num'],
                'entropy': step['entropy']['mean'],
                'verified': True,
                'score': score
            })
        else:
            verifications.append({
                'step_num': step['step_num'],
                'entropy': step['entropy']['mean'],
                'verified': False,
                'score': None
            })
    
    verified_only = [v for v in verifications if v['verified']]
    
    return {
        'strategy': 'random',
        'verifications': verifications,
        'num_verified': len(verified_only),
        'total_steps': len(steps),
        'verification_ratio': len(verified_only) / len(steps),
        'final_decision_correct': all(v['score'] > 0.5 for v in verified_only) == problem_result['is_correct'] if verified_only else False
    }


# ============================================================
# STRATEGY 3: ENTROPY-BASED SELECTIVE (OUR METHOD)
# ============================================================

def entropy_based_verification(problem_result, entropy_threshold=None):
    """
    Verify ONLY the steps with entropy above threshold.
    Low-entropy steps (confident) are skipped.
    High-entropy steps (uncertain) are verified.
    
    This is our novel contribution.
    
    If threshold is None, use mean entropy as threshold (auto-adaptive).
    """
    steps = [s for s in problem_result['steps'] if s['entropy']['num_tokens'] > 0]
    
    if not steps:
        return None
    
    # Determine threshold
    if entropy_threshold is None:
        # Auto-adaptive: use mean entropy across this problem's steps
        entropies = [s['entropy']['mean'] for s in steps]
        entropy_threshold = np.mean(entropies)
    
    verifications = []
    for step in steps:
        if step['entropy']['mean'] > entropy_threshold:
            # High entropy - verify it
            score = simulated_verify_step(step, problem_result['is_correct'])
            verifications.append({
                'step_num': step['step_num'],
                'entropy': step['entropy']['mean'],
                'verified': True,
                'score': score
            })
        else:
            # Low entropy - trust the model
            verifications.append({
                'step_num': step['step_num'],
                'entropy': step['entropy']['mean'],
                'verified': False,
                'score': None
            })
    
    verified_only = [v for v in verifications if v['verified']]
    
    return {
        'strategy': 'entropy_based',
        'entropy_threshold': entropy_threshold,
        'verifications': verifications,
        'num_verified': len(verified_only),
        'total_steps': len(steps),
        'verification_ratio': len(verified_only) / len(steps) if steps else 0,
        'final_decision_correct': all(v['score'] > 0.5 for v in verified_only) == problem_result['is_correct'] if verified_only else True
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare_strategies(model_name, results_path):
    """
    Run all three strategies on a model's results and compare.
    """
    print(f"\n{'='*70}")
    print(f"VERIFICATION STRATEGIES COMPARISON: {model_name}")
    print(f"{'='*70}")
    
    with open(results_path) as f:
        results = json.load(f)
    
    strategy_results = {
        'full': [],
        'random': [],
        'entropy_based': []
    }
    
    # Run each strategy on each problem
    for problem in results:
        full = full_verification(problem)
        rand = random_verification(problem, budget_ratio=0.5)
        ent = entropy_based_verification(problem)
        
        if full:
            strategy_results['full'].append(full)
        if rand:
            strategy_results['random'].append(rand)
        if ent:
            strategy_results['entropy_based'].append(ent)
    
    # Compute aggregate metrics
    print(f"\n{'Strategy':<20} {'Avg Verif':<12} {'Avg Ratio':<12} {'Decision Acc':<15}")
    print("-" * 70)
    
    summary = {}
    for strat_name, strat_results in strategy_results.items():
        if not strat_results:
            continue
            
        avg_verified = np.mean([r['num_verified'] for r in strat_results])
        avg_ratio = np.mean([r['verification_ratio'] for r in strat_results])
        decision_acc = np.mean([r['final_decision_correct'] for r in strat_results])
        
        summary[strat_name] = {
            'avg_verifications_per_problem': float(avg_verified),
            'avg_verification_ratio': float(avg_ratio),
            'decision_accuracy': float(decision_acc),
            'num_problems': len(strat_results)
        }
        
        print(f"{strat_name:<20} {avg_verified:<12.1f} {avg_ratio:<12.2%} {decision_acc:<15.2%}")
    
    # Key insight
    if 'full' in summary and 'entropy_based' in summary:
        cost_savings = 1 - (summary['entropy_based']['avg_verification_ratio'] / summary['full']['avg_verification_ratio'])
        print(f"\n{'='*70}")
        print(f"KEY INSIGHT:")
        print(f"  Entropy-based verification saved {cost_savings*100:.0f}% compute vs full verification")
        print(f"  While maintaining {summary['entropy_based']['decision_accuracy']*100:.0f}% decision accuracy")
        print(f"{'='*70}")
    
    return summary


def main():
    import sys
    
    # Default: run all three models
    model_files = {
        'qwen-math-1.5b': 'results/qwen_math_1.5b_entropy.json',
        'llama-3b': 'results/llama_3b_entropy.json',
        'gemma3-4b': 'results/gemma3_4b_entropy.json'
    }
    
    all_summaries = {}
    for model_name, path in model_files.items():
        if os.path.exists(path):
            summary = compare_strategies(model_name, path)
            all_summaries[model_name] = summary
        else:
            print(f"\nSkipping {model_name} - no results at {path}")
    
    # Save combined summary
    os.makedirs("results", exist_ok=True)
    with open("results/verification_comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    # Final cross-model comparison
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COMPARISON: Entropy-Based Verification")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Verif Ratio':<15} {'Decision Acc':<15} {'Compute Saved':<15}")
    print("-" * 70)
    
    for model_name, summary in all_summaries.items():
        if 'entropy_based' in summary and 'full' in summary:
            eb = summary['entropy_based']
            full = summary['full']
            savings = 1 - (eb['avg_verification_ratio'] / full['avg_verification_ratio']) if full['avg_verification_ratio'] > 0 else 0
            print(f"{model_name:<20} {eb['avg_verification_ratio']:<15.2%} {eb['decision_accuracy']:<15.2%} {savings*100:<15.0f}%")
    
    print(f"\nSaved to: results/verification_comparison.json")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()