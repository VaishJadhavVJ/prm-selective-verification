"""
Professional Results Analysis
==============================
Generates publication-quality charts and tables using pandas + matplotlib.

Outputs:
- analysis/figures/  (PNG charts for paper and presentation)
- analysis/tables/   (CSV tables for reference)

Run: python3 analysis/generate_figures.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors
COLORS = {
    'qwen': '#2196F3',
    'llama': '#FF5722', 
    'gemma': '#4CAF50',
    'correct': '#4CAF50',
    'wrong': '#F44336',
    'high_entropy': '#FF9800',
    'low_entropy': '#2196F3',
}

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('analysis/figures', exist_ok=True)
os.makedirs('analysis/tables', exist_ok=True)


def load_all_results():
    """Load all entropy results into a structured format."""
    files = {
        'Qwen 1.5B': 'results/qwen_math_1.5b_entropy.json',
        'Llama 3B': 'results/llama_3b_entropy.json',
        'Gemma 4B': 'results/gemma3_4b_entropy.json'
    }
    
    all_data = {}
    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                all_data[name] = json.load(f)
    return all_data


def build_problem_dataframe(all_data):
    """Build a pandas DataFrame with one row per problem."""
    rows = []
    for model_name, problems in all_data.items():
        for p in problems:
            rows.append({
                'model': model_name,
                'id': p.get('id', ''),
                'question': p.get('question', '')[:80],
                'expected': p.get('expected_answer', ''),
                'predicted': p.get('model_answer', ''),
                'correct': p.get('is_correct', False),
                'mean_entropy': p.get('mean_entropy', 0),
                'num_steps': p.get('num_steps', 0),
                'num_tokens': p.get('num_tokens_generated', 0),
                'inference_time': p.get('inference_time', 0),
            })
    return pd.DataFrame(rows)


def build_step_dataframe(all_data):
    """Build a pandas DataFrame with one row per reasoning step."""
    rows = []
    for model_name, problems in all_data.items():
        for p in problems:
            for s in p.get('steps', []):
                entropy = s.get('entropy', {})
                if entropy.get('num_tokens', 0) > 0:
                    rows.append({
                        'model': model_name,
                        'problem_id': p.get('id', ''),
                        'correct': p.get('is_correct', False),
                        'step_num': s.get('step_num', 0),
                        'step_text': s.get('text', '')[:100],
                        'mean_entropy': entropy.get('mean', 0),
                        'max_entropy': entropy.get('max', 0),
                        'min_entropy': entropy.get('min', 0),
                        'num_tokens': entropy.get('num_tokens', 0),
                    })
    return pd.DataFrame(rows)


# ============================================================
# FIGURE 1: Model Accuracy Comparison
# ============================================================
def fig1_accuracy_comparison(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    accuracy = df.groupby('model')['correct'].mean() * 100
    accuracy = accuracy.reindex(['Qwen 1.5B', 'Llama 3B', 'Gemma 4B'])
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    
    bars = ax.bar(accuracy.index, accuracy.values, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, accuracy.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy on GSM8K (n=200)')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig1_accuracy_comparison.png')
    plt.close()
    print("  Saved: fig1_accuracy_comparison.png")


# ============================================================
# FIGURE 2: Entropy vs Correctness (the key finding)
# ============================================================
def fig2_entropy_vs_correctness(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    x = np.arange(len(models))
    width = 0.35
    
    correct_means = []
    wrong_means = []
    correct_stds = []
    wrong_stds = []
    
    for m in models:
        model_df = df[df['model'] == m]
        c = model_df[model_df['correct'] == True]['mean_entropy']
        w = model_df[model_df['correct'] == False]['mean_entropy']
        correct_means.append(c.mean())
        wrong_means.append(w.mean())
        correct_stds.append(c.std())
        wrong_stds.append(w.std())
    
    bars1 = ax.bar(x - width/2, correct_means, width, yerr=correct_stds,
                   label='Correct', color=COLORS['correct'], alpha=0.85,
                   capsize=5, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, wrong_means, width, yerr=wrong_stds,
                   label='Wrong', color=COLORS['wrong'], alpha=0.85,
                   capsize=5, edgecolor='white', linewidth=1.5)
    
    ax.set_ylabel('Mean Entropy (bits)')
    ax.set_title('Entropy of Correct vs Wrong Answers')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig2_entropy_vs_correctness.png')
    plt.close()
    print("  Saved: fig2_entropy_vs_correctness.png")


# ============================================================
# FIGURE 3: Step Entropy Distribution (histogram)
# ============================================================
def fig3_entropy_distribution(step_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    
    for ax, model, color in zip(axes, models, colors):
        model_steps = step_df[step_df['model'] == model]['mean_entropy']
        threshold = model_steps.mean()
        
        ax.hist(model_steps, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.3f})')
        
        ax.set_xlabel('Step Entropy (bits)')
        ax.set_title(model)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel('Number of Steps')
    fig.suptitle('Step-Level Entropy Distribution Across Models', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig('analysis/figures/fig3_entropy_distribution.png')
    plt.close()
    print("  Saved: fig3_entropy_distribution.png")


# ============================================================
# FIGURE 4: Compute Savings (the headline chart)
# ============================================================
def fig4_compute_savings(step_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    
    savings = []
    for m in models:
        model_steps = step_df[step_df['model'] == m]['mean_entropy']
        threshold = model_steps.mean()
        low = (model_steps <= threshold).sum()
        total = len(model_steps)
        savings.append(low / total * 100)
    
    bars = ax.bar(models, savings, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    ax.set_ylabel('Steps Skipped (%)')
    ax.set_title('Potential Compute Savings via Entropy-Based Verification')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50% baseline')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig4_compute_savings.png')
    plt.close()
    print("  Saved: fig4_compute_savings.png")


# ============================================================
# FIGURE 5: Entropy vs Step Position
# ============================================================
def fig5_entropy_by_step_position(step_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    
    for model, color in zip(models, colors):
        model_steps = step_df[step_df['model'] == model]
        # Only look at first 8 steps (most common range)
        step_entropy = model_steps[model_steps['step_num'] <= 8].groupby('step_num')['mean_entropy'].mean()
        ax.plot(step_entropy.index, step_entropy.values, marker='o', color=color,
                label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Mean Entropy (bits)')
    ax.set_title('Entropy Across Reasoning Step Positions')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig5_entropy_by_step.png')
    plt.close()
    print("  Saved: fig5_entropy_by_step.png")


# ============================================================
# FIGURE 6: Scatter plot - Entropy vs Problem Difficulty
# ============================================================
def fig6_entropy_vs_steps(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    markers = ['o', 's', '^']
    
    for model, color, marker in zip(models, colors, markers):
        model_df = df[df['model'] == model]
        correct = model_df[model_df['correct'] == True]
        wrong = model_df[model_df['correct'] == False]
        
        ax.scatter(correct['num_steps'], correct['mean_entropy'],
                   color=color, marker=marker, alpha=0.4, s=30, label=f'{model} (correct)')
        ax.scatter(wrong['num_steps'], wrong['mean_entropy'],
                   color=color, marker='x', alpha=0.8, s=50, label=f'{model} (wrong)')
    
    ax.set_xlabel('Number of Reasoning Steps')
    ax.set_ylabel('Mean Entropy (bits)')
    ax.set_title('Problem Complexity vs Model Uncertainty')
    ax.legend(fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig6_entropy_vs_steps.png')
    plt.close()
    print("  Saved: fig6_entropy_vs_steps.png")


# ============================================================
# FIGURE 7: Confidently Wrong Analysis
# ============================================================
def fig7_confidently_wrong(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']
    colors = [COLORS['qwen'], COLORS['llama'], COLORS['gemma']]
    
    categories = ['Correct\n(low entropy)', 'Correct\n(high entropy)', 
                  'Wrong\n(high entropy)\n[CAUGHT]', 'Wrong\n(low entropy)\n[MISSED]']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, (model, color) in enumerate(zip(models, colors)):
        model_df = df[df['model'] == model]
        threshold = model_df['mean_entropy'].mean()
        
        counts = [
            len(model_df[(model_df['correct']) & (model_df['mean_entropy'] <= threshold)]),
            len(model_df[(model_df['correct']) & (model_df['mean_entropy'] > threshold)]),
            len(model_df[(~model_df['correct']) & (model_df['mean_entropy'] > threshold)]),
            len(model_df[(~model_df['correct']) & (model_df['mean_entropy'] <= threshold)]),
        ]
        
        ax.bar(x + i * width, counts, width, label=model, color=color, 
               edgecolor='white', linewidth=1)
    
    ax.set_xlabel('')
    ax.set_ylabel('Number of Problems')
    ax.set_title('Detection Capability: What Entropy-Based Verification Catches and Misses')
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('analysis/figures/fig7_confidently_wrong.png')
    plt.close()
    print("  Saved: fig7_confidently_wrong.png")


# ============================================================
# TABLES
# ============================================================
def generate_tables(df, step_df):
    # Table 1: Model comparison
    summary = df.groupby('model').agg(
        problems=('correct', 'count'),
        accuracy=('correct', 'mean'),
        mean_entropy=('mean_entropy', 'mean'),
        std_entropy=('mean_entropy', 'std'),
        avg_steps=('num_steps', 'mean'),
        avg_time=('inference_time', 'mean'),
    ).round(4)
    summary['accuracy'] = (summary['accuracy'] * 100).round(1)
    summary.to_csv('analysis/tables/table1_model_comparison.csv')
    print("  Saved: table1_model_comparison.csv")
    
    # Table 2: Entropy vs correctness
    rows = []
    for model in ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']:
        m = df[df['model'] == model]
        c = m[m['correct']]['mean_entropy']
        w = m[~m['correct']]['mean_entropy']
        rows.append({
            'Model': model,
            'Correct_Mean': round(c.mean(), 4),
            'Correct_Std': round(c.std(), 4),
            'Wrong_Mean': round(w.mean(), 4),
            'Wrong_Std': round(w.std(), 4),
            'Difference': round(w.mean() - c.mean(), 4),
        })
    pd.DataFrame(rows).to_csv('analysis/tables/table2_entropy_vs_correctness.csv', index=False)
    print("  Saved: table2_entropy_vs_correctness.csv")
    
    # Table 3: Compute savings
    rows = []
    for model in ['Qwen 1.5B', 'Llama 3B', 'Gemma 4B']:
        s = step_df[step_df['model'] == model]['mean_entropy']
        threshold = s.mean()
        low = (s <= threshold).sum()
        high = (s > threshold).sum()
        total = len(s)
        rows.append({
            'Model': model,
            'Total_Steps': total,
            'Low_Entropy': low,
            'High_Entropy': high,
            'Savings_Pct': round(low / total * 100, 1),
        })
    pd.DataFrame(rows).to_csv('analysis/tables/table3_compute_savings.csv', index=False)
    print("  Saved: table3_compute_savings.csv")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\nGenerating professional analysis figures and tables...")
    print("=" * 60)
    
    all_data = load_all_results()
    if not all_data:
        print("No results found. Run entropy_pipeline.py first.")
        return
    
    df = build_problem_dataframe(all_data)
    step_df = build_step_dataframe(all_data)
    
    print(f"\nLoaded: {len(df)} problems, {len(step_df)} steps across {df['model'].nunique()} models")
    
    print("\nGenerating figures...")
    fig1_accuracy_comparison(df)
    fig2_entropy_vs_correctness(df)
    fig3_entropy_distribution(step_df)
    fig4_compute_savings(step_df)
    fig5_entropy_by_step_position(step_df)
    fig6_entropy_vs_steps(df)
    fig7_confidently_wrong(df)
    
    print("\nGenerating tables...")
    generate_tables(df, step_df)
    
    print("\n" + "=" * 60)
    print("DONE! All outputs in analysis/figures/ and analysis/tables/")
    print(f"  7 figures (PNG, 300 DPI)")
    print(f"  3 tables (CSV)")
    print("\nUse these directly in your NeurIPS report and presentation slides.")


if __name__ == "__main__":
    main()