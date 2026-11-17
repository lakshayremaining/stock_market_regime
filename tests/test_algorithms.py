import numpy as np
import pandas as pd
from marketsense.analyzer import analyze_prices
from marketsense.hmm_em import GaussianHMMEM

def synthetic_prices(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    regimes = np.repeat([0, 1, 2], [400, 400, 400])
    means = np.choose(regimes, [0.0008, -0.0007, 0.00005])
    sigmas = np.choose(regimes, [0.010, 0.022, 0.006])
    returns = means + rng.normal(0, sigmas)
    close = 100 * np.cumprod(1 + returns)
    volume = rng.lognormal(13, 0.25, n)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({"Close": close, "High": close * (1 + rng.uniform(0.001, 0.02, n)),
        "Low": close * (1 - rng.uniform(0.001, 0.02, n)), "Volume": volume}, index=idx)

def test_hmm_em_probabilities_and_transitions():
    X = np.random.default_rng(3).normal(size=(500, 3))
    model = GaussianHMMEM(n_states=3, max_iter=20).fit(X)
    posterior = model.predict_proba(X)
    assert posterior.shape == (500, 3)
    assert np.allclose(posterior.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(model.score(X))

def test_end_to_end_analysis():
    result = analyze_prices(synthetic_prices(), ticker="TEST", n_states=4)
    probs = result["regime_probabilities"]
    assert set(probs) == {"Bullish", "Bearish", "Sideways", "High Volatility"}
    assert abs(sum(probs.values()) - 1.0) < 1e-8
    assert result["current_regime"] in probs
    assert result["transition_matrix"].shape == (4, 4)
