# Market Memory: Isolate Directional Noise from Volatility Clustering

An empirical quantitative framework using 25+ years of historical S&P 500 data to contrast market direction memory against market magnitude (volatility) memory.

## Core Empirical Findings

* **Proof A (Direction Has No Memory):** Conditional probabilities of positive market days hover tightly near the historical base rate (~53.8%) regardless of preceding consecutive win streaks (1 to 5+ days). Directional serial correlation is statistically indistinguishable from zero.
* **Proof B (Volatility Has Deep Memory):** Absolute daily returns exhibit strongly positive and highly significant autocorrelation coefficients ($\sim0.25$) extending past 20 trading lags, confirming distinct structural risk regimes.

---

## 🛠️ Step-by-Step Implementation Framework

### Phase 1: Data Acquisition & Preprocessing
* **Historical Data Extraction:** Retrieved daily adjusted closing prices for the S&P 500 index (`^GSPC`) spanning a 20+ year window via the `yfinance` API.
* **Returns Computation:** Transformed raw price data into signed, daily percentage returns representing actual economic direction:
  $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
* **Magnitude Isolation:** Extracted absolute returns $|R_t|$ to completely strip away the direction of the daily price swing and isolate pure magnitude/volatility proxy variables.

### Phase 2: Directional Streak Analysis
* **Base Rate Derivation:** Calculated the unconditional baseline probability of a positive trading day over the complete time-series profile ($\sim53.78\%$).
* **Vectorized Streak Processing:** Implemented a sequential tracking array loop tracking historical multi-day directional momentum strings (calculating consecutive rolling up/down clusters).
* **Conditional Probability Profiling:** Grouped filtered subsets by exact streak lengths ($1, 2, 3, 4, 5+$ days) to calculate empirical next-day directional frequencies:
  $$P(\text{Up Tomorrow} \mid \text{Streak} = N)$$

### Phase 3: Volatility Clustering & Autocorrelation Analytics
* **Risk Horizon Modeling:** Generated a 20-day rolling standard deviation metric, scaled to annualized parameters ($100 \times \sigma_{\text{daily}} \times \sqrt{252}$), to visually track and isolate macro shock clusters (e.g., 2008 GFC, 2020 COVID).
* **Autocorrelation Function (ACF) Face-off:** Extracted the sequential correlation structures of both raw return vectors $R_t$ and absolute magnitude vectors $|R_t|$ across $1$ to $20$ trading lags using `statsmodels.tsa.stattools.acf`.
* **Statistical Boundaries:** Plotted a standard $95\%$ statistical confidence interval boundary ($\pm 2/\sqrt{N}$) across all lags to verify formal mathematical significance.

---

## 🚀 Repository Contents
- `finance_p1.ipynb`: Data engineering infrastructure, pipeline extraction, and target feature transformation modules.
- `finance_p2 (1).ipynb`: Statistical analytics notebook executing conditional probability metrics, rolling volatility distributions, and formal time-series autocorrelation testing.
- `app.py`: Production-ready, interactive Multi-Panel Streamlit dashboard built on vectorized back-end metrics.

## 💻 How to Run Locally
Clone the repository and spin up your local server interface instantly:
```bash
pip install streamlit yfinance pandas numpy statsmodels plotly
streamlit run app.py
