# Entropy-Based Selective PRM Verification for SLMs

## Project Overview
A general framework for selective step verification using Process Reward Models (PRMs) on Small Language Models (SLMs). We use entropy-based uncertainty estimation to identify which reasoning steps are critical and only verify those, improving inference speed without sacrificing accuracy.

## Evaluation Domains
1. **Options Trading Reasoning** (novel contribution) — multi-step financial reasoning involving Black-Scholes pricing, Greeks computation, and risk/reward analysis
2. **GSM8K** (generalizability) — grade-school math word problems

## SRAI Principles
- **Reliability** — uncertainty-guided verification ensures trustworthy outputs
- **Interpretability** — step-level scores reveal where the model struggles
- **Robustness** — accuracy holds under reduced verification strategies

## Algorithms Compared
1. Full PRM verification (baseline)
2. Random step verification
3. Entropy-based selective verification (our method)

## Project Structure
```
prm-project/
├── data/
│   ├── gsm8k/          # GSM8K dataset
│   ├── options/         # Processed options trading scenarios
│   └── raw/             # Raw data from Yahoo Finance
├── scripts/
│   ├── download_gsm8k.py
│   ├── fetch_options_data.py
│   ├── generate_scenarios.py
│   └── verify_steps.py
├── models/              # Model configs and checkpoints
├── results/             # Experiment outputs
├── docs/                # Proposal, reports
└── notebooks/           # Exploration notebooks
```

## Setup
```bash
pip install datasets yfinance pandas numpy scipy
```

## Team
- [Student 1] — Data collection, dataset construction
- [Student 2] — Model pipeline, PRM implementation
