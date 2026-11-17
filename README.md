# MarketSense

MarketSense is a stock market dashboard that studies recent price movement and groups market behaviour into regimes such as bullish, bearish, sideways and high volatility.

## What it shows

- Recent price chart and current market regime
- Regime probabilities and annualized volatility
- A chat panel for questions about the selected stock
- Analysis for supported NSE stocks and NVIDIA

## How it works

The analysis uses a Gaussian Hidden Markov Model trained with Baum–Welch EM. A Bayesian Network combines market features with the inferred regime to produce the displayed probabilities.

The dashboard displays the latest 15 trading sessions. The model uses a longer historical series for fitting so that its estimates are not based on only 15 observations.

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

Open the local URL printed by Vercel.

## API

Send a POST request to `/api/analyze`:

```json
{"ticker":"RELIANCE"}
```

## CLI

```bash
python run.py TCS
```

To analyse a CSV file:

```bash
python run.py TCS --csv sample.csv
```

## Tests

```bash
pip install pytest
pytest -q
```

## Data

Market data is fetched through yfinance. The source may be delayed, unavailable or rate-limited. Model probabilities are estimates based on the available historical data.
