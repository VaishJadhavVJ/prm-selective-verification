"""
Quick test: Run Qwen2.5-Math-1.5B on 5 GSM8K problems.
Saves outputs to results/gsm8k_sample_outputs.json
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading model... (first time downloads ~3GB)")
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!")

    # Load 5 GSM8K test problems
    with open("data/gsm8k/test.json") as f:
        all_problems = json.load(f)
    
    problems = all_problems[:10]
    results = []

    for i, problem in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}: {problem['question'][:100]}...")
        
        prompt = f"Solve this step by step:\n{problem['question']}\nShow your reasoning at each step."
        
        messages = [
            {"role": "system", "content": "You are a math problem solver. Solve problems step by step, showing clear reasoning at each step."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print(f"Model output:\n{response[:300]}...")
        print(f"Expected answer: {problem['final_answer']}")
        
        results.append({
            "id": problem["id"],
            "question": problem["question"],
            "expected_answer": problem["final_answer"],
            "expected_steps": problem["steps"],
            "model_output": response
        })

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/gsm8k_sample_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Saved {len(results)} results to results/gsm8k_sample_outputs.json")
    print("Push to GitHub: git add . && git commit -m 'add sample model outputs' && git push")

if __name__ == "__main__":
    main()
