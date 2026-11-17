from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

REGIMES = ["Bullish", "Bearish", "Sideways", "High Volatility"]
PARENT_STATES = {
    "Trend": ["Down", "Up"],
    "Volatility": ["Low", "Medium", "High"],
    "Momentum": ["Negative", "Neutral", "Positive"],
    "HMMRegime": REGIMES
}

class MarketRegimeBayesianNetwork:
    """Discrete BN: Trend, Volatility, Momentum, HMMRegime -> MarketRegime."""
    def __init__(self):
        self.parent_priors_ = {}
        self.cpt_ = {}

    def fit(self, raw_features: pd.DataFrame, hmm_regimes):
        trend = raw_features["trend"].astype(str).to_numpy()
        vol = raw_features["realized_volatility"].to_numpy()
        mom = raw_features["momentum_20d"].to_numpy()
        hmm = np.asarray(hmm_regimes, dtype=str)
        v1, v2 = np.quantile(vol, [1/3, 2/3])
        m1, m2 = np.quantile(mom, [1/3, 2/3])
        vol_cat = np.where(vol <= v1, "Low", np.where(vol <= v2, "Medium", "High"))
        mom_cat = np.where(mom <= m1, "Negative", np.where(mom <= m2, "Neutral", "Positive"))
        encoded = {"Trend": trend, "Volatility": vol_cat, "Momentum": mom_cat, "HMMRegime": hmm}
        for parent, states in PARENT_STATES.items():
            counts = pd.Series(encoded[parent]).value_counts(normalize=True)
            probs = np.array([counts.get(s, 0.0) for s in states], dtype=float) + 1e-6
            self.parent_priors_[parent] = probs / probs.sum()
        for combo in itertools.product(*(PARENT_STATES[p] for p in PARENT_STATES)):
            t, v, m, h = combo
            scores = {r: 0.15 for r in REGIMES}
            scores[h] += 1.8
            scores["Bullish"] += (0.9 if t == "Up" else 0) + (0.8 if m == "Positive" else 0)
            scores["Bearish"] += (0.9 if t == "Down" else 0) + (0.8 if m == "Negative" else 0)
            scores["Sideways"] += 0.7 if m == "Neutral" else 0
            scores["High Volatility"] += 1.5 if v == "High" else (0.5 if v == "Medium" else 0)
            values = np.array([scores[r] for r in REGIMES], dtype=float)
            values = np.exp(values - values.max())
            self.cpt_[combo] = values / values.sum()
        return self

    def infer(self, evidence: dict):
        required = list(PARENT_STATES)
        missing = [p for p in required if p not in evidence]
        if missing:
            raise ValueError(f"Evidence is missing parent variables: {missing}")
        combo = tuple(evidence[p] for p in required)
        if combo not in self.cpt_:
            raise ValueError(f"Invalid Bayesian evidence: {evidence}")
        posterior = self.cpt_[combo]
        return {regime: float(posterior[i]) for i, regime in enumerate(REGIMES)}

    def latest_evidence(self, raw_features, latest_hmm_regime):
        vol = raw_features["realized_volatility"]
        mom = raw_features["momentum_20d"]
        row = raw_features.iloc[-1]
        v1, v2 = np.quantile(vol, [1/3, 2/3])
        m1, m2 = np.quantile(mom, [1/3, 2/3])
        return {
            "Trend": str(row["trend"]),
            "Volatility": "Low" if row["realized_volatility"] <= v1 else ("Medium" if row["realized_volatility"] <= v2 else "High"),
            "Momentum": "Negative" if row["momentum_20d"] <= m1 else ("Neutral" if row["momentum_20d"] <= m2 else "Positive"),
            "HMMRegime": str(latest_hmm_regime)
        }
