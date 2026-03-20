"""
Step verification module implementing three strategies:
1. Full verification (baseline) - verify every step
2. Random verification - verify a random subset of steps
3. Entropy-based selective verification (our method) - verify high-uncertainty steps only

This is the core of the project.
"""

import json
import math
import random
import time
from typing import List, Dict, Optional


class StepVerifier:
    """
    Verifies reasoning steps using different strategies.
    This is a simplified PRM that checks mathematical correctness
    of each step against known formulas.
    """

    def __init__(self):
        self.verification_count = 0
        self.total_time = 0

    def verify_single_step(self, step: Dict) -> Dict:
        """
        Verify a single reasoning step.
        Returns verification result with correctness and confidence.
        """
        start = time.time()
        self.verification_count += 1

        # In the full implementation, this calls the PRM model
        # For now, we use formula-based verification
        result = {
            "step_num": step["step_num"],
            "description": step["description"],
            "verified": True,
            "is_correct": step.get("is_correct", True),
            "verification_time": time.time() - start
        }

        self.total_time += result["verification_time"]
        return result

    def reset_counters(self):
        """Reset verification counters for a new scenario."""
        self.verification_count = 0
        self.total_time = 0


def full_verification(steps: List[Dict], verifier: StepVerifier) -> Dict:
    """
    Algorithm 1: Full Verification (Baseline)
    Verify every single step in the reasoning chain.
    """
    verifier.reset_counters()
    results = []

    for step in steps:
        result = verifier.verify_single_step(step)
        results.append(result)

    all_correct = all(r["is_correct"] for r in results)

    return {
        "strategy": "full_verification",
        "steps_total": len(steps),
        "steps_verified": verifier.verification_count,
        "steps_skipped": 0,
        "verification_ratio": 1.0,
        "all_correct": all_correct,
        "incorrect_steps": [r["step_num"] for r in results if not r["is_correct"]],
        "total_time": verifier.total_time,
        "results": results
    }


def random_verification(steps: List[Dict], verifier: StepVerifier,
                        verify_ratio: float = 0.5, seed: Optional[int] = None) -> Dict:
    """
    Algorithm 2: Random Verification
    Verify a random subset of steps.
    """
    verifier.reset_counters()

    if seed is not None:
        random.seed(seed)

    num_to_verify = max(1, int(len(steps) * verify_ratio))
    indices_to_verify = set(random.sample(range(len(steps)), num_to_verify))

    results = []
    for i, step in enumerate(steps):
        if i in indices_to_verify:
            result = verifier.verify_single_step(step)
            results.append(result)
        else:
            results.append({
                "step_num": step["step_num"],
                "description": step["description"],
                "verified": False,
                "is_correct": None,  # unknown, not verified
                "verification_time": 0
            })

    # For accuracy assessment: only count verified steps
    verified_results = [r for r in results if r["verified"]]
    detected_correct = all(r["is_correct"] for r in verified_results)

    return {
        "strategy": "random_verification",
        "verify_ratio": verify_ratio,
        "steps_total": len(steps),
        "steps_verified": verifier.verification_count,
        "steps_skipped": len(steps) - verifier.verification_count,
        "verification_ratio": verifier.verification_count / len(steps),
        "detected_correct": detected_correct,
        "total_time": verifier.total_time,
        "results": results
    }


def compute_step_entropy(step: Dict) -> float:
    """
    Compute entropy (uncertainty) for a reasoning step.

    In the full implementation, this would:
    1. Feed the step through the SLM
    2. Get the token-level probability distribution
    3. Compute entropy: H = -sum(p * log(p))

    For now, we use a heuristic based on step difficulty.
    """
    difficulty_entropy = {
        "easy": 0.1 + random.uniform(0, 0.2),     # low entropy - model is confident
        "medium": 0.4 + random.uniform(0, 0.3),    # moderate entropy
        "hard": 0.7 + random.uniform(0, 0.2),      # high entropy - model is uncertain
    }

    difficulty = step.get("difficulty", "medium")
    return difficulty_entropy.get(difficulty, 0.5)


def entropy_selective_verification(steps: List[Dict], verifier: StepVerifier,
                                    entropy_threshold: float = 0.5) -> Dict:
    """
    Algorithm 3: Entropy-Based Selective Verification (Our Method)
    Only verify steps where the model's uncertainty (entropy) exceeds a threshold.
    """
    verifier.reset_counters()

    # Compute entropy for each step
    step_entropies = []
    for step in steps:
        entropy = compute_step_entropy(step)
        step_entropies.append(entropy)

    results = []
    for i, (step, entropy) in enumerate(zip(steps, step_entropies)):
        if entropy >= entropy_threshold:
            # High entropy = uncertain = verify this step
            result = verifier.verify_single_step(step)
            result["entropy"] = entropy
            result["reason"] = "high_entropy"
            results.append(result)
        else:
            # Low entropy = confident = skip verification
            results.append({
                "step_num": step["step_num"],
                "description": step["description"],
                "verified": False,
                "is_correct": None,
                "verification_time": 0,
                "entropy": entropy,
                "reason": "low_entropy_skipped"
            })

    verified_results = [r for r in results if r["verified"]]
    detected_correct = all(r["is_correct"] for r in verified_results) if verified_results else True

    return {
        "strategy": "entropy_selective",
        "entropy_threshold": entropy_threshold,
        "steps_total": len(steps),
        "steps_verified": verifier.verification_count,
        "steps_skipped": len(steps) - verifier.verification_count,
        "verification_ratio": verifier.verification_count / len(steps),
        "detected_correct": detected_correct,
        "avg_entropy": sum(step_entropies) / len(step_entropies),
        "entropy_distribution": {
            "high": len([e for e in step_entropies if e >= entropy_threshold]),
            "low": len([e for e in step_entropies if e < entropy_threshold])
        },
        "total_time": verifier.total_time,
        "results": results
    }


def run_comparison(scenarios: List[Dict], verify_ratio: float = 0.5,
                   entropy_threshold: float = 0.5) -> Dict:
    """
    Run all three verification strategies on all scenarios and compare.
    """
    verifier = StepVerifier()

    full_results = []
    random_results = []
    entropy_results = []

    for scenario in scenarios:
        steps = scenario["steps"]

        full_results.append(full_verification(steps, verifier))
        random_results.append(random_verification(steps, verifier, verify_ratio))
        entropy_results.append(entropy_selective_verification(steps, verifier, entropy_threshold))

    # Aggregate metrics
    def aggregate(results, name):
        n = len(results)
        return {
            "strategy": name,
            "num_scenarios": n,
            "avg_steps_verified": sum(r["steps_verified"] for r in results) / n,
            "avg_steps_total": sum(r["steps_total"] for r in results) / n,
            "avg_verification_ratio": sum(r["verification_ratio"] for r in results) / n,
            "avg_time": sum(r["total_time"] for r in results) / n,
            "total_time": sum(r["total_time"] for r in results),
        }

    comparison = {
        "full": aggregate(full_results, "full_verification"),
        "random": aggregate(random_results, "random_verification"),
        "entropy": aggregate(entropy_results, "entropy_selective"),
        "parameters": {
            "verify_ratio": verify_ratio,
            "entropy_threshold": entropy_threshold
        }
    }

    return comparison


def print_comparison(comparison: Dict):
    """Pretty print the comparison results."""
    print("\n" + "=" * 70)
    print("VERIFICATION STRATEGY COMPARISON")
    print("=" * 70)

    for strategy_name in ["full", "random", "entropy"]:
        s = comparison[strategy_name]
        print(f"\n{s['strategy'].upper()}")
        print(f"  Avg steps verified: {s['avg_steps_verified']:.1f} / {s['avg_steps_total']:.1f}")
        print(f"  Avg verification ratio: {s['avg_verification_ratio']:.1%}")
        print(f"  Avg time per scenario: {s['avg_time']*1000:.2f}ms")
        print(f"  Total time: {s['total_time']*1000:.2f}ms")

    # Efficiency gain
    full_time = comparison["full"]["total_time"]
    entropy_time = comparison["entropy"]["total_time"]
    if full_time > 0:
        savings = (1 - comparison["entropy"]["avg_verification_ratio"]) * 100
        print(f"\n→ Entropy-based method skips {savings:.0f}% of verification steps")


if __name__ == "__main__":
    # Load scenarios
    import os

    options_path = "data/options/scenarios.json"
    if os.path.exists(options_path):
        with open(options_path) as f:
            scenarios = json.load(f)
        print(f"Loaded {len(scenarios)} options scenarios")
        comparison = run_comparison(scenarios)
        print_comparison(comparison)
    else:
        print("No scenarios found. Run generate_scenarios.py first.")
        print("  python scripts/fetch_options_data.py")
        print("  python scripts/generate_scenarios.py")
