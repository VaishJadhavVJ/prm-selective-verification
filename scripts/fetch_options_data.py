"""
Fetch real options chain data from Yahoo Finance for dataset construction.
Pulls stock prices, options chains, and computes key metrics.
"""

import json
import os
from datetime import datetime, timedelta
import math


def install_deps():
    """Install required packages."""
    try:
        import yfinance
        import pandas
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install yfinance pandas --break-system-packages -q")


def fetch_stock_data(ticker):
    """Fetch current stock info and recent price history."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    info = stock.info

    # Get recent price history (30 days)
    hist = stock.history(period="1mo")

    if hist.empty:
        print(f"  Warning: No price history for {ticker}")
        return None

    current_price = hist['Close'].iloc[-1]
    price_30d_ago = hist['Close'].iloc[0] if len(hist) > 0 else current_price
    avg_volume = hist['Volume'].mean()

    return {
        "ticker": ticker,
        "company_name": info.get("longName", ticker),
        "current_price": round(float(current_price), 2),
        "price_30d_ago": round(float(price_30d_ago), 2),
        "price_change_30d_pct": round(float((current_price - price_30d_ago) / price_30d_ago * 100), 2),
        "avg_volume": int(avg_volume),
        "sector": info.get("sector", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "fetched_at": datetime.now().isoformat()
    }


def fetch_options_chain(ticker, max_expiries=3):
    """Fetch options chain data for a ticker."""
    import yfinance as yf
    import pandas as pd

    stock = yf.Ticker(ticker)

    # Get available expiration dates
    try:
        expirations = stock.options
    except Exception as e:
        print(f"  Warning: Could not fetch options for {ticker}: {e}")
        return []

    if not expirations:
        print(f"  Warning: No options available for {ticker}")
        return []

    # Take the nearest expiration dates
    selected_expiries = expirations[:max_expiries]
    chains = []

    for expiry in selected_expiries:
        try:
            opt = stock.option_chain(expiry)

            # Process calls
            for _, row in opt.calls.iterrows():
                chains.append({
                    "type": "call",
                    "expiry": expiry,
                    "strike": float(row["strike"]),
                    "last_price": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    "open_interest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
                    "implied_volatility": float(row.get("impliedVolatility", 0)),
                    "in_the_money": bool(row.get("inTheMoney", False))
                })

            # Process puts
            for _, row in opt.puts.iterrows():
                chains.append({
                    "type": "put",
                    "expiry": expiry,
                    "strike": float(row["strike"]),
                    "last_price": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    "open_interest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
                    "implied_volatility": float(row.get("impliedVolatility", 0)),
                    "in_the_money": bool(row.get("inTheMoney", False))
                })
        except Exception as e:
            print(f"  Warning: Error fetching {expiry} for {ticker}: {e}")

    return chains


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price.
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk-free rate
    sigma: implied volatility
    """
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute option Greeks.
    Returns delta, gamma, theta, vega.
    """
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    # Theta (per day)
    if option_type == "call":
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (per 1% change in volatility)
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4)
    }


# Default tickers to fetch
DEFAULT_TICKERS = [
    "AAPL", "TSLA", "NVDA", "AMZN", "MSFT",
    "SPY", "QQQ", "META", "GOOGL", "AMD"
]

RISK_FREE_RATE = 0.05  # ~current US Treasury rate


def fetch_all(tickers=None, output_dir="data/raw"):
    """Fetch stock and options data for all tickers."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    os.makedirs(output_dir, exist_ok=True)
    install_deps()

    all_data = []

    for ticker in tickers:
        print(f"\nFetching {ticker}...")

        stock_data = fetch_stock_data(ticker)
        if stock_data is None:
            continue

        print(f"  Price: ${stock_data['current_price']}")

        options = fetch_options_chain(ticker)
        print(f"  Options contracts: {len(options)}")

        # Compute theoretical prices and Greeks for each option
        for opt in options:
            # Calculate days to expiry
            expiry_date = datetime.strptime(opt["expiry"], "%Y-%m-%d")
            days_to_expiry = (expiry_date - datetime.now()).days
            T = max(days_to_expiry / 365, 0.001)

            S = stock_data["current_price"]
            K = opt["strike"]
            sigma = opt["implied_volatility"]

            # Theoretical price
            if opt["type"] == "call":
                opt["theoretical_price"] = round(black_scholes_call(S, K, T, RISK_FREE_RATE, sigma), 2)
            else:
                opt["theoretical_price"] = round(black_scholes_put(S, K, T, RISK_FREE_RATE, sigma), 2)

            # Greeks
            opt["greeks"] = compute_greeks(S, K, T, RISK_FREE_RATE, sigma, opt["type"])
            opt["days_to_expiry"] = days_to_expiry

            # Breakeven
            if opt["type"] == "call":
                opt["breakeven"] = round(K + opt["last_price"], 2)
            else:
                opt["breakeven"] = round(K - opt["last_price"], 2)

            # Max loss (for buyer)
            opt["max_loss"] = round(opt["last_price"] * 100, 2)  # per contract

        ticker_data = {
            "stock": stock_data,
            "options": options,
            "meta": {
                "risk_free_rate": RISK_FREE_RATE,
                "num_contracts": len(options),
                "expiries_fetched": len(set(o["expiry"] for o in options))
            }
        }

        all_data.append(ticker_data)

        # Save individual ticker file
        ticker_path = os.path.join(output_dir, f"{ticker}.json")
        with open(ticker_path, "w") as f:
            json.dump(ticker_data, f, indent=2)
        print(f"  Saved to {ticker_path}")

    # Save combined file
    combined_path = os.path.join(output_dir, "all_tickers.json")
    with open(combined_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nAll data saved to {combined_path}")
    print(f"Total tickers: {len(all_data)}")
    print(f"Total option contracts: {sum(len(d['options']) for d in all_data)}")

    return all_data


if __name__ == "__main__":
    fetch_all()
