from __future__ import annotations
import numpy as np
import pandas as pd
from .features import make_features
from .hmm_em import GaussianHMMEM, map_states_to_regimes
from .bayesian import MarketRegimeBayesianNetwork

def analyze_prices(prices: pd.DataFrame, ticker="UNKNOWN", n_states=4):
    raw, X, scaler = make_features(prices)
    hmm = GaussianHMMEM(n_states=n_states, max_iter=150, tol=1e-4, random_state=42).fit(X)
    states, state_mapping, state_summary = map_states_to_regimes(hmm, raw, X)
    hmm_regimes = np.array([state_mapping[int(s)] for s in states])
    raw = raw.copy()
    raw["hidden_state"], raw["hmm_regime"] = states, hmm_regimes
    bn = MarketRegimeBayesianNetwork().fit(raw, hmm_regimes)
    evidence = bn.latest_evidence(raw, hmm_regimes[-1])
    probabilities = bn.infer(evidence)
    current_regime = max(probabilities, key=probabilities.get)
    transitions = pd.DataFrame(hmm.transmat_, index=[f"State {i}" for i in range(n_states)], columns=[f"State {i}" for i in range(n_states)])
    close = prices["Close"].reindex(raw.index)
    annual_vol = float(prices["Close"].pct_change().dropna().std() * np.sqrt(252))
    report = (f"{ticker}: the latest observations are classified as **{current_regime}** "
        f"(Bayesian posterior {probabilities[current_regime]:.1%}). The HMM's most recent hidden state is State {int(states[-1])}, interpreted as {hmm_regimes[-1]}. "
        f"Annualized realized volatility over the available history is {annual_vol:.1%}. This is a classification of observed historical behaviour, not a forecast or investment recommendation.")
    return {"ticker": ticker, "raw_features": raw, "hmm": hmm, "hmm_states": states, "hmm_regimes": hmm_regimes,
        "state_summary": state_summary, "state_mapping": state_mapping, "transition_matrix": transitions,
        "log_likelihood": hmm.log_likelihood_history_[-1], "em_iterations": hmm.n_iter_, "em_converged": hmm.converged_,
        "bayesian_network": bn, "bayesian_evidence": evidence, "regime_probabilities": probabilities,
        "current_regime": current_regime, "annualized_volatility": annual_vol, "close": close, "agent_report": report}
