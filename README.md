# FDT Market Impact Replication Package

This repository contains the manuscript source, simulation scripts, figure-generation pipeline, and the public LOBSTER-based analysis used in:

> Atharva Parande, *A Conditional Fluctuation--Dissipation Identity for Market Impact: Diagnostics, Reconstruction, and an Order-Book Pilot*.

## Repository Contents

- `paper.tex`: manuscript source
- `scripts/generate_plot_data.py`: stylized equilibrium, relaxation, and stress-map data
- `scripts/run_monte_carlo.py`: Monte Carlo size/power and manipulation-check simulations
- `scripts/run_operational_validation.py`: rolling synthetic validation outputs
- `scripts/run_lobster_empirical.py`: public LOBSTER real-data analysis for AAPL, GOOG, and AMZN
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

## Python Requirements

- Python 3.11+
- `matplotlib`
- `numpy`

The Monte Carlo, synthetic validation, and LOBSTER analysis scripts otherwise use only the Python standard library.

## Data Notes

The real-data section uses the public LOBSTER sample files included in this repository for June 21, 2012. Any future extension using licensed TAQ or larger vendor datasets would remain subject to the relevant data-provider terms and is therefore not bundled here.
