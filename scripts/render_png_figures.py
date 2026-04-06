from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figures" / "data"
OUT_DIR = ROOT / "figures" / "generated"

BLUE = "#1d4ed8"
TEAL = "#0f766e"
GOLD = "#c77d12"
RED = "#b42318"
NAVY = "#0d1b2a"
GRAY = "#6b7280"
LIGHT = "#dbe2ea"
BG = "#f8fafc"

HEATMAP = LinearSegmentedColormap.from_list(
    "fdt_heat",
    ["#1f6aa5", "#78b8d8", "#f7f7f7", "#f7c97a", "#b42318"],
)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": "white",
            "savefig.facecolor": BG,
            "axes.edgecolor": LIGHT,
            "axes.labelcolor": NAVY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
            "grid.color": "#e5e7eb",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
        }
    )


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_fdt_signature() -> None:
    rows = read_csv(DATA_DIR / "equilibrium_curves.csv")
    omega = np.array([row["omega"] for row in rows], dtype=float)
    im_g = np.array([row["im_g"] for row in rows], dtype=float)
    sp_eq = np.array([row["sp_eq"] for row in rows], dtype=float)
    sp_stress = np.array([row["sp_stress"] for row in rows], dtype=float)
    theta_proxy = float(np.median(omega * sp_eq / (2.0 * np.maximum(im_g, 1e-12))))
    rhs = omega * sp_eq / (2.0 * theta_proxy)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(omega, im_g, color=BLUE, lw=2.4, label=r"$\Im \widehat{G}(i\omega)$")
    ax.plot(omega, rhs, color=GOLD, lw=2.2, ls="--", label=r"$\frac{\omega}{2\Theta} S_P^{\mathrm{eq}}(\omega)$")
    ax.set_title("Equilibrium Signature")
    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_ylabel("Normalized scale")
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="lower left", fontsize=10.5)

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(omega, sp_eq, color=TEAL, lw=2.4, label="Equilibrium spectrum")
    ax.plot(omega, sp_stress, color=RED, lw=2.2, label="Stressed spectrum")
    ax.set_title("Volatility Regimes")
    ax.set_xlabel(r"Frequency $\omega$")
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="lower left", fontsize=10.5)

    finish(fig, OUT_DIR / "fdt_signature.png")


def render_relaxation() -> None:
    modes = read_csv(DATA_DIR / "relaxation_modes.csv")
    kernel = read_csv(DATA_DIR / "kernel_decay.csv")
    rho = np.array([row["rho"] for row in modes], dtype=float)
    weights = np.array([row["weight"] for row in modes], dtype=float)
    t = np.array([row["t"] for row in kernel], dtype=float)
    full_kernel = np.array([row["kernel"] for row in kernel], dtype=float)
    reduced = np.array([row["reduced"] for row in kernel], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.bar(rho, weights, width=np.maximum(rho * 0.16, 0.08), color=BLUE, alpha=0.9)
    ax.set_xscale("log")
    ax.set_title("Sparse Relaxation Spectrum")
    ax.set_xlabel(r"Decay rate $\rho$")
    ax.set_ylabel("Weight")
    ax.grid(True, axis="y", alpha=0.35)

    ax = axes[1]
    ax.plot(t, full_kernel, color=BLUE, lw=2.4, label="Full kernel")
    ax.plot(t, reduced, color=GOLD, lw=2.2, ls="--", label="Reduced approximation")
    ax.set_title("Kernel Compression")
    ax.set_xlabel(r"Lag $t$")
    ax.set_ylabel(r"$G(t)$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")

    finish(fig, OUT_DIR / "relaxation_spectrum.png")


def render_heatmap() -> None:
    rows = read_csv(DATA_DIR / "stress_heatmap.csv")
    x_vals = sorted({row["log10_omega"] for row in rows})
    y_vals = sorted({row["regime"] for row in rows})
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    x_index = {x: i for i, x in enumerate(x_vals)}
    y_index = {y: i for i, y in enumerate(y_vals)}
    for row in rows:
        grid[y_index[row["regime"]], x_index[row["log10_omega"]]] = row["delta"]

    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=HEATMAP)
    ax.set_title("Stylized Frequency-by-Regime Stress Map")
    ax.set_xlabel(r"$\log_{10}(\omega)$")
    ax.set_ylabel("Regime index")
    step_x = max(1, len(x_vals) // 6)
    ax.set_xticks(range(0, len(x_vals), step_x))
    ax.set_xticklabels([f"{x_vals[i]:.1f}" for i in range(0, len(x_vals), step_x)])
    step_y = max(1, len(y_vals) // 6)
    ax.set_yticks(range(0, len(y_vals), step_y))
    ax.set_yticklabels([str(int(y_vals[i])) for i in range(0, len(y_vals), step_y)])
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\Delta(\omega)$")

    finish(fig, OUT_DIR / "stress_heatmap.png")


def render_mc_delta() -> None:
    rows = read_csv(DATA_DIR / "monte_carlo_delta_example.csv")
    omega = np.array([row["omega"] for row in rows], dtype=float)
    true_delta = np.array([row["true_delta"] for row in rows], dtype=float)
    estimated = np.array([row["estimated_delta"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.set_xscale("log")
    ax.plot(omega, true_delta, color=RED, lw=2.4, label=r"True $\Delta(\omega)$")
    ax.plot(omega, estimated, color=BLUE, lw=2.2, ls="--", label=r"Estimated $\widehat{\Delta}(\omega)$")
    ax.axhline(0.0, color=GRAY, lw=1.0, alpha=0.8)
    ax.set_title(r"Representative Monte Carlo Deviation at $n = 2^{14}$")
    ax.set_xlabel(r"Frequency $\omega$")
    ax.set_ylabel(r"$\Delta(\omega)$")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right")

    finish(fig, OUT_DIR / "monte_carlo_delta.png")


def render_operational_validation() -> None:
    sessions = read_csv(DATA_DIR / "operational_validation_sessions.csv")
    heatmap = read_csv(DATA_DIR / "operational_validation_heatmap.csv")

    sess = np.array([row["session"] for row in sessions], dtype=float)
    theta = np.array([row["theta_hat"] for row in sessions], dtype=float)
    stat = np.array([row["test_stat"] for row in sessions], dtype=float)
    crit = np.array([row["critical_value"] for row in sessions], dtype=float)

    x_vals = sorted({row["session"] for row in heatmap})
    y_vals = sorted({row["log10_omega"] for row in heatmap})
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    x_index = {x: i for i, x in enumerate(x_vals)}
    y_index = {y: i for i, y in enumerate(y_vals)}
    for row in heatmap:
        grid[y_index[row["log10_omega"]], x_index[row["session"]]] = row["delta_hat"]

    fig = plt.figure(figsize=(12.6, 7.6), constrained_layout=True)
    gs = GridSpec(2, 2, height_ratios=[1, 1.18], figure=fig)
    ax_theta = fig.add_subplot(gs[0, 0])
    ax_stat = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, :])

    for ax in (ax_theta, ax_stat):
        ax.axvspan(1, 20, color="#eaf3ff", alpha=0.95)
        ax.axvspan(20, 40, color="#fff1db", alpha=0.95)
        ax.axvspan(40, 60, color="#fee8e7", alpha=0.95)

    ax_theta.plot(sess, theta, color=NAVY, lw=2.4)
    ax_theta.axhline(float(np.median(theta[:20])), color=GOLD, ls="--", lw=1.8)
    ax_theta.set_title(r"Rolling Thermal Index $\widehat{\Theta}$")
    ax_theta.set_xlabel("Session")
    ax_theta.grid(True, alpha=0.35)

    ax_stat.plot(sess, stat, color=RED, lw=2.4)
    ax_stat.plot(sess, crit, color=GRAY, ls="--", lw=1.8)
    ax_stat.set_title(r"Test Statistic vs Critical Value")
    ax_stat.set_xlabel("Session")
    ax_stat.grid(True, alpha=0.35)

    im = ax_map.imshow(grid, aspect="auto", origin="lower", cmap=HEATMAP)
    ax_map.set_title(r"Session-by-Frequency Map of $\widehat{\Delta}(\omega)$")
    ax_map.set_xlabel("Session")
    ax_map.set_ylabel(r"$\log_{10}(\omega)$")
    ax_map.set_xticks(np.linspace(0, len(x_vals) - 1, 6))
    ax_map.set_xticklabels([f"{int(v)}" for v in np.linspace(min(x_vals), max(x_vals), 6)])
    ax_map.set_yticks(np.linspace(0, len(y_vals) - 1, 5))
    ax_map.set_yticklabels([f"{v:.1f}" for v in np.linspace(min(y_vals), max(y_vals), 5)])
    cbar = fig.colorbar(im, ax=ax_map, pad=0.01)
    cbar.set_label(r"$\widehat{\Delta}(\omega)$")

    finish(fig, OUT_DIR / "operational_validation.png")


def render_lobster_empirical() -> None:
    curves = read_csv(DATA_DIR / "lobster_empirical_curves.csv")
    rolling = read_csv(DATA_DIR / "lobster_empirical_theta_roll.csv")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in curves:
        grouped[str(row["window"])].append(row)

    fig = plt.figure(figsize=(14.2, 8.6), constrained_layout=True)
    gs = GridSpec(2, 3, height_ratios=[1, 1.05], figure=fig)
    windows = ["Open", "Midday", "Close"]
    for idx, name in enumerate(windows):
        ax = fig.add_subplot(gs[0, idx])
        rows = grouped[name]
        omega = np.array([row["omega"] for row in rows], dtype=float)
        im_g = np.array([row["im_g_hat"] for row in rows], dtype=float)
        rhs = np.array([row["fdt_rhs"] for row in rows], dtype=float)
        scale = max(np.max(im_g), np.max(rhs), 1e-12)
        line1, = ax.plot(omega, im_g / scale, color=BLUE, lw=2.2, label=r"Normalized $\Im \widehat{G}$")
        line2, = ax.plot(omega, rhs / scale, color=RED, lw=2.0, ls="--", label="Normalized FDT benchmark")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"Frequency $\omega$")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.35)
        if idx == 0:
            ax.set_ylabel("Normalized level")
            ax.legend(loc="upper left", fontsize=10.0)

    ax = fig.add_subplot(gs[1, :])
    time_sec = np.array([row["time_sec"] for row in rolling], dtype=float)
    theta = np.array([row["theta_hat"] for row in rolling], dtype=float)
    reject = np.array([row["reject_5pct"] for row in rolling], dtype=float)
    ax.plot(time_sec, theta, color=NAVY, lw=2.4)
    ax.scatter(time_sec[reject > 0.5], theta[reject > 0.5], color=RED, s=36, zorder=3, label="Reject at 5%")
    ax.set_title(r"Rolling Thermal Index Through the Day")
    ax.set_xlabel("Seconds after midnight")
    ax.set_ylabel(r"$\widehat{\Theta}$")
    ax.grid(True, alpha=0.35)
    if np.any(reject > 0.5):
        ax.legend(loc="upper left", fontsize=10.5)

    finish(fig, OUT_DIR / "lobster_empirical.png")


def render_realdata_breakdown() -> None:
    rows = read_csv(DATA_DIR / "lobster_panel_summary.csv")
    symbols = ["AAPL", "GOOG", "AMZN"]
    windows = ["Open", "Midday", "Close"]
    ratio = np.full((len(symbols), len(windows)), np.nan)
    theta = np.full((len(symbols), len(windows)), np.nan)
    reject = np.zeros((len(symbols), len(windows)))

    s_index = {s: i for i, s in enumerate(symbols)}
    w_index = {w: i for i, w in enumerate(windows)}
    for row in rows:
        s = str(row["symbol"])
        w = str(row["window"])
        if s in s_index and w in w_index:
            i = s_index[s]
            j = w_index[w]
            ratio[i, j] = float(row["deviation_ratio"])
            theta[i, j] = float(row["theta_hat"])
            reject[i, j] = float(row["reject_5pct"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)

    im = axes[0].imshow(ratio, cmap=HEATMAP, aspect="auto", vmin=0.0, vmax=max(1.2, float(np.nanmax(ratio))))
    axes[0].set_title("Deviation Ratio by Symbol and Window")
    axes[0].set_xticks(range(len(windows)))
    axes[0].set_xticklabels(windows)
    axes[0].set_yticks(range(len(symbols)))
    axes[0].set_yticklabels(symbols)
    for i in range(len(symbols)):
        for j in range(len(windows)):
            text = f"{ratio[i,j]:.2f}"
            if reject[i, j] > 0.5:
                text += "\nreject"
            axes[0].text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=11,
                color=NAVY,
                fontweight="bold" if reject[i, j] > 0.5 else None,
            )
            if reject[i, j] > 0.5:
                axes[0].add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=RED, linewidth=2.4))
    cbar = fig.colorbar(im, ax=axes[0], pad=0.02)
    cbar.set_label("Deviation ratio")

    for symbol in symbols:
        i = s_index[symbol]
        axes[1].plot(windows, theta[i], marker="o", lw=2.2, label=symbol)
    axes[1].set_title(r"Thermal Index Across the Same Trading Day", fontsize=12)
    axes[1].set_ylabel(r"$\widehat{\Theta}$")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=10.0,
    )

    finish(fig, OUT_DIR / "realdata_breakdown.png")


def render_state_map() -> None:
    mc = read_csv(DATA_DIR / "operational_validation_summary.csv")
    aapl = read_csv(DATA_DIR / "lobster_empirical_summary.csv")

    fig, ax = plt.subplots(figsize=(9.8, 6.2), constrained_layout=True)

    regime_style = {
        "equilibrium": ("Synthetic equilibrium", TEAL, "o", 140),
        "transition": ("Synthetic transition", GOLD, "s", 150),
        "stress": ("Synthetic stress", RED, "^", 170),
    }

    for row in mc:
        regime = str(row["regime"])
        label, color, marker, size = regime_style[regime]
        ax.scatter(
            float(row["avg_theta_hat"]),
            float(row["avg_peak_abs_delta"]),
            s=size,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        ax.text(
            float(row["avg_theta_hat"]) + 0.015,
            float(row["avg_peak_abs_delta"]) + 0.03,
            regime.capitalize(),
            fontsize=9,
            color=color,
        )

    aapl_style = {
        "Open": (NAVY, "o"),
        "Midday": ("#7c2d12", "D"),
        "Close": ("#4b5563", "P"),
    }
    for row in aapl:
        window = str(row["window"])
        color, marker = aapl_style[window]
        ax.scatter(
            float(row["theta_hat"]),
            float(row["deviation_ratio"]),
            s=140,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.95,
            label=f"AAPL {window}",
            zorder=4,
        )
        ax.text(
            float(row["theta_hat"]) + 0.05,
            float(row["deviation_ratio"]) + 0.025,
            f"AAPL {window}",
            fontsize=9,
            color=color,
        )

    ax.axhspan(0.0, 0.6, color="#ecfdf5", alpha=0.85, zorder=0)
    ax.axhspan(0.6, 1.2, color="#fffbeb", alpha=0.9, zorder=0)
    ax.axhspan(1.2, 3.0, color="#fef2f2", alpha=0.92, zorder=0)
    ax.text(0.15, 0.18, "Near-equilibrium band", color=TEAL, fontsize=9)
    ax.text(0.15, 0.82, "Transitional band", color=GOLD, fontsize=9)
    ax.text(0.15, 2.15, "Breakdown band", color=RED, fontsize=9)

    ax.set_title(r"Market-State Map: Thermal Index vs FDT Deviation")
    ax.set_xlabel(r"Thermal index / average $\widehat{\Theta}$")
    ax.set_ylabel(r"Deviation magnitude")
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(0.0, 3.0)
    ax.grid(True, alpha=0.28)

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    finish(fig, OUT_DIR / "market_state_map.png")


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_fdt_signature()
    render_relaxation()
    render_heatmap()
    render_mc_delta()
    render_operational_validation()
    render_lobster_empirical()
    render_realdata_breakdown()
    render_state_map()
    print(f"Rendered matplotlib figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
