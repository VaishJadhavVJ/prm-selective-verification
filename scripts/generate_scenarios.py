"""
Generate options trading reasoning scenarios from raw data.
Each scenario includes a question, multi-step reasoning chain,
and step-level correctness labels for PRM verification.
"""

import json
import os
import random


def load_raw_data(raw_dir="data/raw"):
    """Load all fetched ticker data."""
    combined_path = os.path.join(raw_dir, "all_tickers.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            return json.load(f)
    return []


def generate_basic_call_scenario(stock, option):
    """
    Generate a basic call option analysis scenario.
    Type: Single option evaluation
    Complexity: Basic
    """
    S = stock["current_price"]
    K = option["strike"]
    premium = option["last_price"]
    days = option["days_to_expiry"]
    iv = option["implied_volatility"]
    greeks = option["greeks"]

    intrinsic = max(S - K, 0)
    time_value = round(premium - intrinsic, 2)
    breakeven = round(K + premium, 2)
    pct_to_breakeven = round((breakeven - S) / S * 100, 2)
    max_loss = round(premium * 100, 2)
    moneyness = "in-the-money" if S > K else "out-of-the-money" if S < K else "at-the-money"

    question = (
        f"{stock['ticker']} is currently trading at ${S}. "
        f"You are considering buying a call option with a ${K} strike price, "
        f"expiring in {days} days, with a premium of ${premium}. "
        f"Implied volatility is {round(iv * 100, 1)}%. "
        f"Should you buy this call? Walk through the analysis step by step."
    )

    steps = [
        {
            "step_num": 1,
            "description": "Determine moneyness",
            "reasoning": f"The current stock price is ${S} and the strike price is ${K}. "
                        f"Since {'the stock price is above the strike' if S > K else 'the stock price is below the strike' if S < K else 'they are equal'}, "
                        f"this call is {moneyness}.",
            "result": moneyness,
            "correct_result": moneyness,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"Compare S=${S} vs K=${K}"
        },
        {
            "step_num": 2,
            "description": "Calculate intrinsic value",
            "reasoning": f"Intrinsic value of a call = max(Stock Price - Strike Price, 0) = max(${S} - ${K}, 0) = ${intrinsic}.",
            "result": intrinsic,
            "correct_result": intrinsic,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"max({S} - {K}, 0) = {intrinsic}"
        },
        {
            "step_num": 3,
            "description": "Calculate time value",
            "reasoning": f"Time value = Premium - Intrinsic Value = ${premium} - ${intrinsic} = ${time_value}.",
            "result": time_value,
            "correct_result": time_value,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"{premium} - {intrinsic} = {time_value}"
        },
        {
            "step_num": 4,
            "description": "Calculate breakeven price",
            "reasoning": f"Breakeven at expiration = Strike + Premium = ${K} + ${premium} = ${breakeven}. "
                        f"The stock needs to rise {pct_to_breakeven}% from the current price.",
            "result": breakeven,
            "correct_result": breakeven,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"{K} + {premium} = {breakeven}; ({breakeven}-{S})/{S}*100 = {pct_to_breakeven}%"
        },
        {
            "step_num": 5,
            "description": "Analyze Greeks - Delta",
            "reasoning": f"Delta is {greeks['delta']:.4f}, meaning for every $1 increase in {stock['ticker']}'s price, "
                        f"this option's value changes by approximately ${abs(greeks['delta']):.2f}.",
            "result": greeks["delta"],
            "correct_result": greeks["delta"],
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Black-Scholes delta calculation"
        },
        {
            "step_num": 6,
            "description": "Analyze Greeks - Theta",
            "reasoning": f"Theta is {greeks['theta']:.4f}, meaning this option loses approximately "
                        f"${abs(greeks['theta']):.2f} per day from time decay alone. "
                        f"Over {days} days, that is significant{'ly destructive' if days < 14 else ''}.",
            "result": greeks["theta"],
            "correct_result": greeks["theta"],
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Black-Scholes theta calculation"
        },
        {
            "step_num": 7,
            "description": "Determine max loss",
            "reasoning": f"Maximum loss when buying a call is the total premium paid. "
                        f"Per contract (100 shares): ${premium} x 100 = ${max_loss}.",
            "result": max_loss,
            "correct_result": max_loss,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"{premium} * 100 = {max_loss}"
        },
        {
            "step_num": 8,
            "description": "Overall assessment",
            "reasoning": generate_assessment(S, K, premium, days, iv, greeks, breakeven, pct_to_breakeven, moneyness),
            "result": "assessment",
            "correct_result": "assessment",
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Composite judgment based on prior steps"
        }
    ]

    return {
        "question": question,
        "steps": steps,
        "num_steps": len(steps),
        "final_answer": {
            "breakeven": breakeven,
            "max_loss": max_loss,
            "moneyness": moneyness,
            "delta": greeks["delta"],
            "theta": greeks["theta"]
        },
        "scenario_type": "basic_call_analysis",
        "difficulty": "basic"
    }


def generate_basic_put_scenario(stock, option):
    """
    Generate a basic put option analysis scenario.
    Type: Single option evaluation
    Complexity: Basic
    """
    S = stock["current_price"]
    K = option["strike"]
    premium = option["last_price"]
    days = option["days_to_expiry"]
    iv = option["implied_volatility"]
    greeks = option["greeks"]

    intrinsic = max(K - S, 0)
    time_value = round(premium - intrinsic, 2)
    breakeven = round(K - premium, 2)
    pct_to_breakeven = round((S - breakeven) / S * 100, 2)
    max_loss = round(premium * 100, 2)
    moneyness = "in-the-money" if K > S else "out-of-the-money" if K < S else "at-the-money"

    question = (
        f"{stock['ticker']} is currently trading at ${S}. "
        f"You are considering buying a put option with a ${K} strike price, "
        f"expiring in {days} days, with a premium of ${premium}. "
        f"Implied volatility is {round(iv * 100, 1)}%. "
        f"Should you buy this put? Walk through the analysis step by step."
    )

    steps = [
        {
            "step_num": 1,
            "description": "Determine moneyness",
            "reasoning": f"The current stock price is ${S} and the strike price is ${K}. "
                        f"For a put, it is in-the-money when strike > stock price. "
                        f"Since {'K > S' if K > S else 'K < S' if K < S else 'K = S'}, this put is {moneyness}.",
            "result": moneyness,
            "correct_result": moneyness,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"Put ITM check: K=${K} vs S=${S}"
        },
        {
            "step_num": 2,
            "description": "Calculate intrinsic value",
            "reasoning": f"Intrinsic value of a put = max(Strike Price - Stock Price, 0) = max(${K} - ${S}, 0) = ${intrinsic}.",
            "result": intrinsic,
            "correct_result": intrinsic,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"max({K} - {S}, 0) = {intrinsic}"
        },
        {
            "step_num": 3,
            "description": "Calculate time value",
            "reasoning": f"Time value = Premium - Intrinsic Value = ${premium} - ${intrinsic} = ${time_value}.",
            "result": time_value,
            "correct_result": time_value,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"{premium} - {intrinsic} = {time_value}"
        },
        {
            "step_num": 4,
            "description": "Calculate breakeven price",
            "reasoning": f"Breakeven at expiration = Strike - Premium = ${K} - ${premium} = ${breakeven}. "
                        f"The stock needs to fall {pct_to_breakeven}% from the current price.",
            "result": breakeven,
            "correct_result": breakeven,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"{K} - {premium} = {breakeven}; ({S}-{breakeven})/{S}*100 = {pct_to_breakeven}%"
        },
        {
            "step_num": 5,
            "description": "Analyze Greeks - Delta",
            "reasoning": f"Delta is {greeks['delta']:.4f} (negative for puts), meaning for every $1 increase in "
                        f"{stock['ticker']}'s price, this option loses approximately ${abs(greeks['delta']):.2f}.",
            "result": greeks["delta"],
            "correct_result": greeks["delta"],
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Black-Scholes delta calculation"
        },
        {
            "step_num": 6,
            "description": "Analyze Greeks - Theta",
            "reasoning": f"Theta is {greeks['theta']:.4f}, meaning this option loses approximately "
                        f"${abs(greeks['theta']):.2f} per day from time decay.",
            "result": greeks["theta"],
            "correct_result": greeks["theta"],
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Black-Scholes theta calculation"
        },
        {
            "step_num": 7,
            "description": "Determine max loss and max profit",
            "reasoning": f"Maximum loss when buying a put is the premium: ${premium} x 100 = ${max_loss} per contract. "
                        f"Maximum profit is if the stock goes to $0: (${K} - $0 - ${premium}) x 100 = ${round((K - premium) * 100, 2)} per contract.",
            "result": max_loss,
            "correct_result": max_loss,
            "is_correct": True,
            "difficulty": "easy",
            "verification_formula": f"Max loss: {premium} * 100 = {max_loss}"
        },
        {
            "step_num": 8,
            "description": "Overall assessment",
            "reasoning": generate_put_assessment(S, K, premium, days, iv, greeks, breakeven, pct_to_breakeven, moneyness),
            "result": "assessment",
            "correct_result": "assessment",
            "is_correct": True,
            "difficulty": "hard",
            "verification_formula": "Composite judgment based on prior steps"
        }
    ]

    return {
        "question": question,
        "steps": steps,
        "num_steps": len(steps),
        "final_answer": {
            "breakeven": breakeven,
            "max_loss": max_loss,
            "moneyness": moneyness,
            "delta": greeks["delta"],
            "theta": greeks["theta"]
        },
        "scenario_type": "basic_put_analysis",
        "difficulty": "basic"
    }


def generate_assessment(S, K, premium, days, iv, greeks, breakeven, pct_to_breakeven, moneyness):
    """Generate a textual assessment of the call option."""
    signals = []

    if pct_to_breakeven > 10:
        signals.append(f"The stock needs to move {pct_to_breakeven}% to break even, which is aggressive")
    elif pct_to_breakeven > 5:
        signals.append(f"A {pct_to_breakeven}% move to breakeven is moderately challenging")
    else:
        signals.append(f"Only a {pct_to_breakeven}% move is needed to break even, which is achievable")

    if days < 14:
        signals.append(f"with only {days} days to expiry, theta decay is accelerating rapidly")
    elif days < 30:
        signals.append(f"{days} days provides moderate time for the thesis to play out")
    else:
        signals.append(f"{days} days gives ample time for the move")

    if iv > 0.4:
        signals.append("high implied volatility means the premium is expensive")
    elif iv < 0.2:
        signals.append("low implied volatility makes this relatively cheap")

    return f"Considering all factors: {'; '.join(signals)}. This is a {'speculative' if pct_to_breakeven > 8 else 'moderate' if pct_to_breakeven > 4 else 'conservative'} bullish position."


def generate_put_assessment(S, K, premium, days, iv, greeks, breakeven, pct_to_breakeven, moneyness):
    """Generate a textual assessment of the put option."""
    signals = []

    if pct_to_breakeven > 10:
        signals.append(f"The stock needs to fall {pct_to_breakeven}% to break even, which is a large move")
    elif pct_to_breakeven > 5:
        signals.append(f"A {pct_to_breakeven}% decline to breakeven is moderately challenging")
    else:
        signals.append(f"Only a {pct_to_breakeven}% decline is needed to break even")

    if days < 14:
        signals.append(f"with only {days} days to expiry, theta decay is accelerating rapidly")
    elif days < 30:
        signals.append(f"{days} days provides moderate time for the bearish thesis")
    else:
        signals.append(f"{days} days gives ample time for the move")

    return f"Considering all factors: {'; '.join(signals)}. This is a {'speculative' if pct_to_breakeven > 8 else 'moderate' if pct_to_breakeven > 4 else 'conservative'} bearish position."


def generate_all_scenarios(raw_data, output_dir="data/options"):
    """Generate scenarios from all fetched data."""
    os.makedirs(output_dir, exist_ok=True)

    all_scenarios = []

    for ticker_data in raw_data:
        stock = ticker_data["stock"]
        options = ticker_data["options"]

        if not options:
            continue

        # Filter for options with reasonable data
        valid_options = [
            o for o in options
            if o["last_price"] > 0
            and o["implied_volatility"] > 0
            and o["days_to_expiry"] > 0
        ]

        # Sample a mix of calls and puts, ITM and OTM
        calls = [o for o in valid_options if o["type"] == "call"]
        puts = [o for o in valid_options if o["type"] == "put"]

        # Take up to 5 calls and 5 puts per ticker, mix of ITM/OTM
        selected_calls = []
        itm_calls = [c for c in calls if c["in_the_money"]]
        otm_calls = [c for c in calls if not c["in_the_money"]]
        selected_calls.extend(random.sample(itm_calls, min(2, len(itm_calls))))
        selected_calls.extend(random.sample(otm_calls, min(3, len(otm_calls))))

        selected_puts = []
        itm_puts = [p for p in puts if p["in_the_money"]]
        otm_puts = [p for p in puts if not p["in_the_money"]]
        selected_puts.extend(random.sample(itm_puts, min(2, len(itm_puts))))
        selected_puts.extend(random.sample(otm_puts, min(3, len(otm_puts))))

        for opt in selected_calls:
            scenario = generate_basic_call_scenario(stock, opt)
            scenario["id"] = f"options_{len(all_scenarios):04d}"
            scenario["domain"] = "options_trading"
            scenario["ticker"] = stock["ticker"]
            all_scenarios.append(scenario)

        for opt in selected_puts:
            scenario = generate_basic_put_scenario(stock, opt)
            scenario["id"] = f"options_{len(all_scenarios):04d}"
            scenario["domain"] = "options_trading"
            scenario["ticker"] = stock["ticker"]
            all_scenarios.append(scenario)

    # Save all scenarios
    output_path = os.path.join(output_dir, "scenarios.json")
    with open(output_path, "w") as f:
        json.dump(all_scenarios, f, indent=2)

    # Save stats
    stats = {
        "total_scenarios": len(all_scenarios),
        "call_scenarios": len([s for s in all_scenarios if s["scenario_type"] == "basic_call_analysis"]),
        "put_scenarios": len([s for s in all_scenarios if s["scenario_type"] == "basic_put_analysis"]),
        "tickers_covered": list(set(s["ticker"] for s in all_scenarios)),
        "avg_steps": sum(s["num_steps"] for s in all_scenarios) / max(len(all_scenarios), 1),
        "difficulty_distribution": {
            "basic": len([s for s in all_scenarios if s["difficulty"] == "basic"])
        }
    }
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nGenerated {len(all_scenarios)} scenarios")
    print(f"  Calls: {stats['call_scenarios']}")
    print(f"  Puts: {stats['put_scenarios']}")
    print(f"  Tickers: {', '.join(stats['tickers_covered'])}")
    print(f"  Avg steps per scenario: {stats['avg_steps']:.1f}")
    print(f"Saved to {output_path}")

    return all_scenarios


if __name__ == "__main__":
    raw_data = load_raw_data()
    if not raw_data:
        print("No raw data found. Run fetch_options_data.py first.")
        print("  python scripts/fetch_options_data.py")
    else:
        generate_all_scenarios(raw_data)
