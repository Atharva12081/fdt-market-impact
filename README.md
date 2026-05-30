# FDT Market Impact Replication Package

This repository contains the manuscript source, simulation scripts, figure-generation pipeline, and the public LOBSTER-based analysis used in:

> Atharva Parande, *A Conditional Fluctuation--Dissipation Identity for Market Impact: Diagnostics, Reconstruction, and Order-Book Evidence*.

## Repository Contents

- `paper.tex`: manuscript source
- `scripts/generate_plot_data.py`: stylized equilibrium, relaxation, and stress-map data
- `scripts/run_monte_carlo.py`: Monte Carlo size/power and manipulation-check simulations
- `scripts/run_operational_validation.py`: rolling synthetic validation outputs
- `scripts/run_lobster_empirical.py`: public LOBSTER real-data analysis; the retained reported panel is AAPL, GOOG, and AMZN
- `scripts/run_recent_market_proxy.py`: optional current OHLCV proxy screen using public 5-minute bars
- `scripts/render_png_figures.py`: matplotlib figure renderer
- `figures/data/`: generated numerical outputs used for tables and figures
- `figures/generated/`: final PNG figures included in the manuscript
- `data/raw/lobster/`: public LOBSTER sample files used in the real-data section

## Reproducing the Main Outputs

From the repository root:

```bash
python3 scripts/generate_plot_data.py
python3 scripts/run_monte_carlo.py
python3 scripts/run_operational_validation.py
python3 scripts/run_lobster_empirical.py
python3 scripts/render_png_figures.py
```

The current-data proxy screen is optional because it requires network access and `yfinance`:

```bash
python3 scripts/run_recent_market_proxy.py
```

## Python Requirements

- Python 3.11+
- `matplotlib`
- `numpy`
- `yfinance` for the optional recent public-bar proxy screen

The Monte Carlo, synthetic validation, and LOBSTER analysis scripts otherwise use only the Python standard library.

## Data Notes

The real-data section uses the public LOBSTER sample files included in this repository for June 21, 2012. The raw folder also includes MSFT and INTC samples; under the retained-band positivity filter used by the estimator, those two names do not produce valid reported windows.

The recent proxy screen writes `figures/data/recent_market_proxy_results.csv` and `figures/data/recent_market_proxy_summary.txt`. The current generated output covers U.S. names over May 15--29, 2026 and Indian names over May 18--29, 2026. It is reported as proxy-tier evidence only because public OHLCV bars do not include true signed executions.

Any future extension using licensed TAQ or larger vendor datasets would remain subject to the relevant data-provider terms and is therefore not bundled here.
