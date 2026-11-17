# MarketSense — Stock Market Regime Analyzer

A dashboard + chat-style interface for classifying the historical behaviour of NSE stocks using up to five years of daily data from Yahoo Finance.

## Models
- **Gaussian Hidden Markov Model (HMM):** four latent states with diagonal Gaussian emissions.
- **Baum–Welch EM:** forward-backward E-step and parameter updates for initial probabilities, transitions, means and variances.
- **Bayesian Network:** Trend, volatility, momentum and HMM regime are used as evidence for the final regime posterior.
- **Features:** daily return, 20-day realized volatility, 20-day momentum and volume change.

The Bayesian Network's parent priors are estimated from the data. Its child conditional probability table currently uses explicit domain assumptions rather than labelled training outcomes. Posterior values should therefore be treated as model estimates, not calibrated real-world probabilities.

## Run locally
Use Python 3.12.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install -g vercel
vercel dev
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
npm install -g vercel
vercel dev
```

Open the local URL printed by Vercel. Search for symbols such as `RELIANCE`, `TCS` or `INFY`. NSE symbols are automatically suffixed with `.NS`.

## Deploy to Vercel
1. Import this GitHub repository into Vercel.
2. Select **Other** as the framework preset.
3. Leave the build command and output directory empty.
4. Deploy. Vercel serves `index.html` and runs `api/analyze.py` as a Python serverless function.

## API
Send `POST /api/analyze` with JSON:
```json
{"ticker":"RELIANCE"}
```

## CLI
```bash
python run.py TCS
```

Optional CSV input must contain `Date`, `Close`, `High`, `Low` and `Volume` columns:
```bash
python run.py TCS --csv sample.csv
```

## Tests
```bash
pip install pytest
pytest -q
```

## Notes
Yahoo Finance through yfinance is an unofficial data source and may be rate-limited. The project is educational: it classifies observed historical behaviour and does not predict exact prices or provide investment advice.
