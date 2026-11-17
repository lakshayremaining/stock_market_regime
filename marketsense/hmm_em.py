from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import logsumexp

class GaussianHMMEM:
    """Gaussian HMM with diagonal covariance trained using Baum-Welch EM."""
    def __init__(self, n_states=4, max_iter=150, tol=1e-4, min_covar=1e-3, random_state=42):
        if n_states < 2:
            raise ValueError("n_states must be at least 2.")
        self.n_states, self.max_iter, self.tol = n_states, max_iter, tol
        self.min_covar, self.random_state = min_covar, random_state

    def _log_emissions(self, X):
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (self.n_features_ * np.log(2 * np.pi)
            + np.log(self.vars_).sum(axis=1)[None, :]
            + (diff ** 2 / self.vars_[None, :, :]).sum(axis=2))

    def _forward_backward(self, log_b):
        T, K = log_b.shape
        log_a = np.log(np.clip(self.transmat_, 1e-300, None))
        log_pi = np.log(np.clip(self.startprob_, 1e-300, None))
        alpha = np.empty((T, K))
        alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            alpha[t] = log_b[t] + logsumexp(alpha[t-1][:, None] + log_a, axis=0)
        beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(log_a + log_b[t+1][None, :] + beta[t+1][None, :], axis=1)
        log_likelihood = float(logsumexp(alpha[-1]))
        log_gamma = alpha + beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            log_xi = alpha[t][:, None] + log_a + log_b[t+1][None, :] + beta[t+1][None, :] - log_likelihood
            xi_sum += np.exp(log_xi)
        return gamma, xi_sum, log_likelihood

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < self.n_states * 10:
            raise ValueError("X must be a 2D array with enough observations for the selected states.")
        if not np.isfinite(X).all():
            raise ValueError("HMM input contains NaN or infinite values.")
        T, self.n_features_ = X.shape
        centered = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        order = np.argsort(centered @ vt[0])
        chunks = np.array_split(order, self.n_states)
        self.means_ = np.vstack([X[c].mean(axis=0) for c in chunks])
        global_var = np.maximum(X.var(axis=0), self.min_covar)
        self.vars_ = np.vstack([np.maximum(X[c].var(axis=0), self.min_covar) if len(c) > 1 else global_var for c in chunks])
        self.startprob_ = np.full(self.n_states, 1 / self.n_states)
        self.transmat_ = np.full((self.n_states, self.n_states), 0.05 / max(self.n_states-1, 1))
        np.fill_diagonal(self.transmat_, 0.95)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
        self.log_likelihood_history_ = []
        previous = -np.inf
        for _ in range(self.max_iter):
            gamma, xi_sum, ll = self._forward_backward(self._log_emissions(X))
            self.log_likelihood_history_.append(ll)
            self.startprob_ = np.maximum(gamma[0], 1e-8)
            self.startprob_ /= self.startprob_.sum()
            trans = xi_sum + 1e-8
            self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
            weights = gamma.sum(axis=0) + 1e-10
            self.means_ = (gamma.T @ X) / weights[:, None]
            for k in range(self.n_states):
                diff = X - self.means_[k]
                self.vars_[k] = np.maximum((gamma[:, k][:, None] * diff**2).sum(axis=0) / weights[k], self.min_covar)
            if np.isfinite(previous) and abs(ll - previous) <= self.tol * (1 + abs(previous)):
                break
            previous = ll
        self.n_iter_ = len(self.log_likelihood_history_)
        self.converged_ = self.n_iter_ < self.max_iter or (len(self.log_likelihood_history_) > 1 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) <= self.tol * (1 + abs(self.log_likelihood_history_[-2])))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        gamma, _, _ = self._forward_backward(self._log_emissions(X))
        return gamma

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        _, _, ll = self._forward_backward(self._log_emissions(np.asarray(X, dtype=float)))
        return ll

def map_states_to_regimes(model, raw_features, X):
    states = model.predict(X)
    rows = []
    for state in range(model.n_states):
        mask = states == state
        rows.append({"state": state, "observations": int(mask.sum()),
            "mean_daily_return": float(raw_features.loc[raw_features.index[mask], "daily_return"].mean()) if mask.any() else 0.0,
            "mean_annualized_volatility": float(raw_features.loc[raw_features.index[mask], "realized_volatility"].mean()) if mask.any() else 0.0})
    summary = pd.DataFrame(rows)
    high_vol = int(summary.sort_values("mean_annualized_volatility").iloc[-1]["state"])
    others = summary[summary.state != high_vol].sort_values("mean_daily_return")
    labels = {}
    if len(others) >= 3:
        labels[int(others.iloc[0].state)] = "Bearish"
        labels[int(others.iloc[len(others)//2].state)] = "Sideways"
        labels[int(others.iloc[-1].state)] = "Bullish"
        for _, row in others.iloc[1:-1].iterrows():
            labels.setdefault(int(row.state), "Sideways")
    else:
        for i, row in enumerate(others.itertuples()):
            labels[int(row.state)] = ["Bearish", "Bullish"][min(i, 1)]
    labels[high_vol] = "High Volatility"
    summary["regime"] = summary["state"].map(labels)
    return states, labels, summary
