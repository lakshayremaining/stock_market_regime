import argparse
import json
import pandas as pd
from marketsense.data import fetch_history
from marketsense.analyzer import analyze_prices

def main():
    parser = argparse.ArgumentParser(description="MarketSense probabilistic market regime analyzer")
    parser.add_argument("ticker", help="NSE ticker, e.g. RELIANCE, TCS, INFY")
    parser.add_argument("--csv", help="Optional local CSV with Date, Close, High, Low, Volume columns")
    parser.add_argument("--states", type=int, default=4, help="Number of HMM hidden states (default: 4)")
    args = parser.parse_args()
    if args.csv:
        prices = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date").sort_index()
        ticker = args.ticker.upper()
    else:
        prices = fetch_history(args.ticker)
        ticker = prices.attrs.get("ticker", args.ticker.upper())
    result = analyze_prices(prices, ticker=ticker, n_states=args.states)
    print("\\n" + "=" * 68)
    print(f"MARKETSENSE REPORT — {ticker}")
    print("=" * 68)
    print(result["agent_report"])
    print("\\nRegime probabilities:")
    for regime, probability in sorted(result["regime_probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {regime:18s} {probability:7.2%}")
    print("\\nHMM / EM diagnostics:")
    print(f"  Log-likelihood: {result['log_likelihood']:.3f}")
    print(f"  EM iterations:  {result['em_iterations']}")
    print(f"  Converged:      {result['em_converged']}")
    print("\\nState interpretation:")
    print(result["state_summary"].to_string(index=False))
    print("\\nTransition matrix:")
    print(result["transition_matrix"].round(3).to_string())
    print("\\nBayesian evidence:")
    print(json.dumps(result["bayesian_evidence"], indent=2))

if __name__ == "__main__":
    main()
