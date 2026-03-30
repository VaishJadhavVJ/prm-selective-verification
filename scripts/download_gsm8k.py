"""
Download and process GSM8K dataset for PRM verification experiments.
GSM8K contains grade-school math problems with step-by-step solutions.
"""

import json
import os

def download_gsm8k():
    """Download GSM8K from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets --break-system-packages -q")
        from datasets import load_dataset

    print("Downloading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    return dataset


def parse_steps(solution_text):
    """
    Parse a GSM8K solution into individual reasoning steps.
    GSM8K solutions use \\n to separate steps, with the final answer after ####.
    """
    # Split answer from solution
    parts = solution_text.split("####")
    solution_body = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else None

    # Split into individual steps
    steps = [s.strip() for s in solution_body.split("\n") if s.strip()]

    return {
        "steps": steps,
        "num_steps": len(steps),
        "final_answer": final_answer
    }


def process_and_save(dataset, output_dir="data/gsm8k"):
    """Process GSM8K into our standard format and save."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        processed = []

        for i, example in enumerate(split_data):
            parsed = parse_steps(example["answer"])

            processed.append({
                "id": f"gsm8k_{split_name}_{i}",
                "domain": "math",
                "question": example["question"],
                "steps": parsed["steps"],
                "num_steps": parsed["num_steps"],
                "final_answer": parsed["final_answer"],
                "raw_solution": example["answer"]
            })

        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w") as f:
            json.dump(processed, f, indent=2)

        print(f"Saved {len(processed)} {split_name} examples to {output_path}")

        # Print a sample
        if processed:
            sample = processed[0]
            print(f"\n--- Sample {split_name} example ---")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Steps ({sample['num_steps']}):")
            for j, step in enumerate(sample['steps']):
                print(f"  Step {j+1}: {step[:80]}...")
            print(f"Answer: {sample['final_answer']}")

    # Save dataset stats
    stats = {
        "train_size": len(dataset["train"]),
        "test_size": len(dataset["test"]),
        "avg_steps_train": sum(
            len(parse_steps(ex["answer"])["steps"]) for ex in dataset["train"]
        ) / len(dataset["train"]),
        "avg_steps_test": sum(
            len(parse_steps(ex["answer"])["steps"]) for ex in dataset["test"]
        ) / len(dataset["test"]),
    }
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDataset stats saved to {stats_path}")
    print(f"  Train: {stats['train_size']} examples, avg {stats['avg_steps_train']:.1f} steps")
    print(f"  Test: {stats['test_size']} examples, avg {stats['avg_steps_test']:.1f} steps")


if __name__ == "__main__":
    dataset = download_gsm8k()
    process_and_save(dataset)
    print("\nGSM8K ready!")
