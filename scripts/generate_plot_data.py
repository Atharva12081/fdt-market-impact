from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figures" / "data"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def logspace(start_exp: float, stop_exp: float, count: int) -> list[float]:
    if count == 1:
        return [10 ** start_exp]
    step = (stop_exp - start_exp) / (count - 1)
    return [10 ** (start_exp + i * step) for i in range(count)]


def build_equilibrium_curves() -> None:
    theta = 0.42
    modes = [(0.18, 0.7), (1.1, 1.4), (4.2, 0.85)]
    rows: list[dict[str, float]] = []
    for omega in logspace(-2, 2, 240):
        im_g = sum(weight * omega * rho / (rho * rho + omega * omega) for rho, weight in modes)
        sp_eq = (2 * theta / omega) * im_g
        sp_stress = sp_eq * (1.0 + 0.22 * math.exp(-((math.log10(omega) - 0.55) ** 2) / 0.22))
        rows.append(
            {
                "omega": omega,
                "im_g": im_g,
                "sp_eq": sp_eq,
                "sp_stress": sp_stress,
            }
        )
    write_csv(DATA_DIR / "equilibrium_curves.csv", ["omega", "im_g", "sp_eq", "sp_stress"], rows)


def build_relaxation_modes() -> None:
    mode_rows = [
        {"rho": 0.15, "weight": 0.62},
        {"rho": 0.85, "weight": 1.35},
        {"rho": 3.8, "weight": 0.78},
        {"rho": 11.5, "weight": 0.34},
    ]
    write_csv(DATA_DIR / "relaxation_modes.csv", ["rho", "weight"], mode_rows)

    decay_rows: list[dict[str, float]] = []
    for t in [i * 0.1 for i in range(0, 181)]:
        kernel = sum(row["weight"] * math.exp(-row["rho"] * t) for row in mode_rows)
        reduced = (
            0.68 * math.exp(-0.22 * t)
            + 1.12 * math.exp(-1.45 * t)
            + 0.41 * math.exp(-7.8 * t)
        )
        decay_rows.append({"t": t, "kernel": kernel, "reduced": reduced})
    write_csv(DATA_DIR / "kernel_decay.csv", ["t", "kernel", "reduced"], decay_rows)


def build_stress_heatmap() -> None:
    rows: list[dict[str, float]] = []
    freqs = logspace(-2, 2, 28)
    for regime in range(12):
        center = 0.25 + 0.11 * regime
        width = max(0.14, 0.42 - 0.018 * regime)
        amplitude = 0.12 + 0.055 * regime
        for omega in freqs:
            x = math.log10(omega)
            base = amplitude * math.exp(-((x - center) ** 2) / width)
            ripple = 0.045 * math.sin(2.7 * x + regime * 0.42)
            delta = base + ripple - 0.06
            rows.append({"log10_omega": x, "regime": regime, "delta": delta})
    write_csv(DATA_DIR / "stress_heatmap.csv", ["log10_omega", "regime", "delta"], rows)


def main() -> None:
    ensure_dirs()
    build_equilibrium_curves()
    build_relaxation_modes()
    build_stress_heatmap()
    print(f"Wrote plot data to {DATA_DIR}")


if __name__ == "__main__":
    main()
