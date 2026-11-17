from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json, sys, traceback
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20000:
                return self.send_json(400, {"error": "Invalid request body."})
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            ticker = str(payload.get("ticker", "")).strip().upper()
            if not ticker:
                return self.send_json(400, {"error": "Enter a stock ticker, e.g. RELIANCE or TCS."})
            from marketsense.data import fetch_history
            from marketsense.analyzer import analyze_prices
            prices = fetch_history(ticker, period="5y")
            symbol = prices.attrs.get("ticker", ticker)
            result = analyze_prices(prices, ticker=symbol, n_states=4)
            raw, close = result["raw_features"], result["close"]
            chart_frame = pd.DataFrame({"close": close, "regime": raw["hmm_regime"]}, index=raw.index)
            monthly = chart_frame.resample("M").last().dropna()
            history = [{"date": date.strftime("%Y-%m"), "close": round(float(row["close"]), 2), "regime": str(row["regime"])} for date, row in monthly.iterrows()]
            state_summary = result["state_summary"].to_dict(orient="records")
            transition = {str(idx): {str(col): round(float(value), 4) for col, value in row.items()} for idx, row in result["transition_matrix"].iterrows()}
            response = {"ticker": symbol, "period": "5 years, daily data", "observations": int(len(prices)),
                "current_price": round(float(prices["Close"].iloc[-1]), 2), "last_updated": prices.index[-1].strftime("%Y-%m-%d"),
                "current_regime": result["current_regime"], "annualized_volatility": round(result["annualized_volatility"], 4),
                "probabilities": {k: round(float(v), 6) for k, v in result["regime_probabilities"].items()},
                "history": history, "state_summary": state_summary, "transition_matrix": transition,
                "bayesian_evidence": result["bayesian_evidence"],
                "hmm": {"states": 4, "log_likelihood": round(result["log_likelihood"], 3), "em_iterations": result["em_iterations"], "em_converged": result["em_converged"]},
                "agent_report": result["agent_report"]}
            self.send_json(200, response)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"error": f"Analysis failed ({type(exc).__name__}): {str(exc)[:220]}"})

    def do_GET(self):
        self.send_json(405, {"error": "Send a POST request to this endpoint."})
