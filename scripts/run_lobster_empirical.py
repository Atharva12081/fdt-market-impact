from __future__ import annotations

import csv
import cmath
import math
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "lobster" / "sample"
RAW_MULTI_DIR = ROOT / "data" / "raw" / "lobster" / "sample_multi"
DATA_DIR = ROOT / "figures" / "data"

START_SEC = 34200.0
END_SEC = 57600.0
BIN_SECONDS = 60.0
BOOT_REPS = 1000
ETA_SCALE = float(os.environ.get("ETA_SCALE", "1.0"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def averaged_exponential(rng: random.Random, m: int) -> float:
    total = 0.0
    for _ in range(m):
        total += rng.expovariate(1.0)
    return total / m


def dft(series: list[float]) -> list[complex]:
    n = len(series)
    out: list[complex] = []
    for j in range(n):
        total = 0j
        for t, value in enumerate(series):
            total += value * cmath.exp(-2j * math.pi * j * t / n)
        out.append(total / math.sqrt(n))
    return out


def smooth(values: list[complex], j: int, m: int) -> complex:
    left = max(0, j - m)
    right = min(len(values), j + m + 1)
    total = 0j
    for idx in range(left, right):
        total += values[idx]
    return total / (right - left)


def load_binned_series(message_path: Path, book_path: Path) -> tuple[list[float], list[float], list[float]]:
    n_bins = int((END_SEC - START_SEC) / BIN_SECONDS)
    signed_volume = [0.0 for _ in range(n_bins)]
    last_mid = [None for _ in range(n_bins)]

    with message_path.open() as msg_f, book_path.open() as book_f:
        msg_reader = csv.reader(msg_f)
        book_reader = csv.reader(book_f)
        current_mid = None
        for msg_row, book_row in zip(msg_reader, book_reader):
            time_sec = float(msg_row[0])
            if time_sec < START_SEC or time_sec >= END_SEC:
                continue
            bin_idx = int((time_sec - START_SEC) // BIN_SECONDS)
            ask = float(book_row[0]) / 10000.0
            bid = float(book_row[2]) / 10000.0
            current_mid = 0.5 * (ask + bid)
            last_mid[bin_idx] = current_mid

            event_type = int(msg_row[1])
            size = float(msg_row[3])
            direction = float(msg_row[5])
            if event_type in (4, 5):
                # LOBSTER direction is the resting limit-order side; retain the
                # raw dataset sign so the reported outputs are reproducible.
                signed_volume[bin_idx] += direction * size

    first_mid = next((value for value in last_mid if value is not None), None)
    if first_mid is None:
        raise RuntimeError("No midpoint observations found in sample.")

    mids: list[float] = []
    current = first_mid
    for value in last_mid:
        if value is not None:
            current = value
        mids.append(current)

    log_mids = [math.log(value) for value in mids]
    returns = [log_mids[i] - log_mids[i - 1] for i in range(1, len(log_mids))]
    flows = signed_volume[1:]
    times = [START_SEC + BIN_SECONDS * i for i in range(1, len(log_mids))]
    return times, returns, flows


def analyze_window(name: str, returns: list[float], flows: list[float], rng: random.Random) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    n = len(returns)
    if n < 64:
        raise RuntimeError(f"Window {name} is too short for spectral analysis.")

    r_mean = sum(returns) / n
    q_mean = sum(flows) / n
    r = [x - r_mean for x in returns]
    q = [x - q_mean for x in flows]

    r_fft = dft(r)
    q_fft = dft(q)
    i_rr = [value * value.conjugate() for value in r_fft]
    i_qq = [value * value.conjugate() for value in q_fft]
    i_rq = [r_fft[j] * q_fft[j].conjugate() for j in range(n)]

    m = max(2, int(round(math.sqrt(n) / 4.0)))
    freq_rows: list[dict[str, float | str]] = []
    theta_candidates: list[float] = []

    retained_indices: list[int] = []
    delta_by_j: dict[int, float] = {}
    im_g_by_j: dict[int, float] = {}
    s_p_by_j: dict[int, float] = {}
    omega_by_j: dict[int, float] = {}

    for j in range(1, n // 2):
        omega = 2.0 * math.pi * j / n
        s_rr = smooth(i_rr, j, m).real
        s_qq = max(smooth(i_qq, j, m).real, 1e-12)
        s_rq = smooth(i_rq, j, m)
        eta = ETA_SCALE * 0.01 * max(s_qq, 1e-12)
        g_hat = s_rq / (s_qq + eta)
        im_g = -g_hat.imag
        if im_g <= 0.0 or s_rr <= 0.0:
            continue
        if j <= 2 or j >= n // 6:
            continue
        theta = omega * s_rr / (2.0 * im_g)
        if theta <= 0.0 or theta > 10.0:
            continue
        theta_candidates.append(theta)
        retained_indices.append(j)
        im_g_by_j[j] = im_g
        s_p_by_j[j] = s_rr
        omega_by_j[j] = omega

    if len(theta_candidates) < 3:
        raise RuntimeError(f"Window {name} retained too few positive frequencies.")

    theta_candidates.sort()
    theta_hat = theta_candidates[len(theta_candidates) // 2]

    for j in retained_indices:
        omega = omega_by_j[j]
        im_g = im_g_by_j[j]
        s_rr = s_p_by_j[j]
        delta = im_g - omega * s_rr / (2.0 * theta_hat)
        delta_by_j[j] = delta

    boot_stats: list[float] = []
    boot_deltas: dict[int, list[float]] = {j: [] for j in retained_indices}
    for _ in range(BOOT_REPS):
        stat = 0.0
        deltas_star: dict[int, float] = {}
        for j in retained_indices:
            omega = omega_by_j[j]
            im_g = im_g_by_j[j]
            s_rr0 = 2.0 * theta_hat * im_g / omega
            s_rr_star = s_rr0 * averaged_exponential(rng, m)
            delta_star = im_g - omega * s_rr_star / (2.0 * theta_hat)
            deltas_star[j] = delta_star
            boot_deltas[j].append(delta_star)
        variances = {}
        for j in retained_indices:
            samples = boot_deltas[j]
            mean = sum(samples) / len(samples)
            variances[j] = max(sum((x - mean) ** 2 for x in samples) / max(len(samples) - 1, 1), 1e-12)
            stat += (deltas_star[j] ** 2) / variances[j]
        boot_stats.append(stat)

    variances = {}
    test_stat = 0.0
    for j in retained_indices:
        samples = boot_deltas[j]
        mean = sum(samples) / len(samples)
        variances[j] = max(sum((x - mean) ** 2 for x in samples) / max(len(samples) - 1, 1), 1e-12)
        test_stat += (delta_by_j[j] ** 2) / variances[j]

    boot_stats_sorted = sorted(boot_stats)
    crit_idx = min(len(boot_stats_sorted) - 1, int(math.ceil(0.95 * len(boot_stats_sorted))) - 1)
    critical_value = boot_stats_sorted[crit_idx]
    p_value = sum(1 for value in boot_stats if value >= test_stat) / len(boot_stats)
    norm_num = math.sqrt(sum(delta_by_j[j] ** 2 for j in retained_indices))
    norm_den = math.sqrt(sum(im_g_by_j[j] ** 2 for j in retained_indices))
    deviation_ratio = norm_num / norm_den if norm_den > 0 else 0.0

    for j in retained_indices:
        freq_rows.append(
            {
                "window": name,
                "omega": round(omega_by_j[j], 8),
                "im_g_hat": round(im_g_by_j[j], 8),
                "fdt_rhs": round(omega_by_j[j] * s_p_by_j[j] / (2.0 * theta_hat), 8),
                "delta_hat": round(delta_by_j[j], 8),
            }
        )

    summary = {
        "window": name,
        "n_obs": n,
        "m": m,
        "eta_scale": round(ETA_SCALE, 3),
        "theta_hat": round(theta_hat, 6),
        "test_stat": round(test_stat, 4),
        "critical_value": round(critical_value, 4),
        "p_value": round(p_value, 4),
        "deviation_ratio": round(deviation_ratio, 4),
        "reject_5pct": int(test_stat > critical_value),
    }
    return summary, freq_rows


def rolling_theta(times: list[float], returns: list[float], flows: list[float], rng: random.Random) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    window_bins = 90
    step = 15
    for start in range(0, len(returns) - window_bins + 1, step):
        end = start + window_bins
        label = f"roll_{start}"
        try:
            summary, _ = analyze_window(label, returns[start:end], flows[start:end], rng)
        except RuntimeError:
            continue
        center_time = times[start + window_bins // 2]
        rows.append(
            {
                "time_sec": round(center_time, 3),
                "theta_hat": summary["theta_hat"],
                "deviation_ratio": summary["deviation_ratio"],
                "reject_5pct": summary["reject_5pct"],
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_symbol(symbol: str, message_path: Path, book_path: Path, rng: random.Random) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, float]], list[dict[str, float | str]]]:
    times, returns, flows = load_binned_series(message_path, book_path)

    total_bins = len(returns)
    windows = [
        ("Open", 0, 90),
        ("Midday", 120, 300),
        ("Close", 300, total_bins),
    ]

    summaries: list[dict[str, float | str]] = []
    freq_rows: list[dict[str, float | str]] = []
    failed_windows: list[str] = []
    for name, start, end in windows:
        try:
            summary, rows = analyze_window(name, returns[start:end], flows[start:end], rng)
        except RuntimeError:
            failed_windows.append(name)
            continue
        summary["symbol"] = symbol
        summaries.append(summary)
        for row in rows:
            row["symbol"] = symbol
        freq_rows.extend(rows)

    if not summaries:
        raise RuntimeError(f"{symbol} retained no valid windows.")

    rolling_rows = rolling_theta(times, returns, flows, rng)
    for row in rolling_rows:
        row["symbol"] = symbol

    return summaries, freq_rows, rolling_rows, [{
        "symbol": symbol,
        "avg_theta_hat": round(sum(float(row["theta_hat"]) for row in summaries) / len(summaries), 6),
        "max_deviation_ratio": round(max(float(row["deviation_ratio"]) for row in summaries), 6),
        "reject_count": int(sum(int(row["reject_5pct"]) for row in summaries)),
        "valid_windows": len(summaries),
        "failed_windows": ",".join(failed_windows),
    }]


def main() -> None:
    ensure_dirs()
    rng = random.Random(20260405)
    aapl_message = RAW_DIR / "AAPL_2012-06-21_34200000_57600000_message_1.csv"
    aapl_book = RAW_DIR / "AAPL_2012-06-21_34200000_57600000_orderbook_1.csv"
    summaries, freq_rows, rolling_rows, _ = analyze_symbol("AAPL", aapl_message, aapl_book, rng)

    write_csv(
        DATA_DIR / "lobster_empirical_summary.csv",
        ["symbol", "window", "n_obs", "m", "eta_scale", "theta_hat", "test_stat", "critical_value", "p_value", "deviation_ratio", "reject_5pct"],
        summaries,
    )
    write_csv(
        DATA_DIR / "lobster_empirical_curves.csv",
        ["symbol", "window", "omega", "im_g_hat", "fdt_rhs", "delta_hat"],
        freq_rows,
    )
    write_csv(
        DATA_DIR / "lobster_empirical_theta_roll.csv",
        ["symbol", "time_sec", "theta_hat", "deviation_ratio", "reject_5pct"],
        rolling_rows,
    )

    with (DATA_DIR / "lobster_empirical_summary.txt").open("w") as f:
        f.write("AAPL LOBSTER empirical evidence (1-minute bins)\n")
        for row in summaries:
            f.write(
                f"{row['window']}: theta_hat={row['theta_hat']}, T={row['test_stat']}, "
                f"crit={row['critical_value']}, p={row['p_value']}, "
                f"ratio={row['deviation_ratio']}, reject={row['reject_5pct']}\n"
            )

    panel_specs = [
        ("AAPL", aapl_message, aapl_book),
        (
            "MSFT",
            RAW_MULTI_DIR / "MSFT_2012-06-21_34200000_57600000_message_1.csv",
            RAW_MULTI_DIR / "MSFT_2012-06-21_34200000_57600000_orderbook_1.csv",
        ),
        (
            "GOOG",
            RAW_MULTI_DIR / "GOOG_2012-06-21_34200000_57600000_message_1.csv",
            RAW_MULTI_DIR / "GOOG_2012-06-21_34200000_57600000_orderbook_1.csv",
        ),
        (
            "AMZN",
            RAW_MULTI_DIR / "AMZN_2012-06-21_34200000_57600000_message_1.csv",
            RAW_MULTI_DIR / "AMZN_2012-06-21_34200000_57600000_orderbook_1.csv",
        ),
        (
            "INTC",
            RAW_MULTI_DIR / "INTC_2012-06-21_34200000_57600000_message_1.csv",
            RAW_MULTI_DIR / "INTC_2012-06-21_34200000_57600000_orderbook_1.csv",
        ),
    ]
    panel_rows: list[dict[str, float | str]] = []
    panel_agg_rows: list[dict[str, float | str]] = []
    for symbol, message_path, book_path in panel_specs:
        try:
            symbol_summaries, _, _, agg_rows = analyze_symbol(symbol, message_path, book_path, rng)
        except RuntimeError:
            continue
        panel_rows.extend(symbol_summaries)
        panel_agg_rows.extend(agg_rows)

    write_csv(
        DATA_DIR / "lobster_panel_summary.csv",
        ["symbol", "window", "n_obs", "m", "eta_scale", "theta_hat", "test_stat", "critical_value", "p_value", "deviation_ratio", "reject_5pct"],
        panel_rows,
    )
    write_csv(
        DATA_DIR / "lobster_panel_aggregate.csv",
        ["symbol", "avg_theta_hat", "max_deviation_ratio", "reject_count", "valid_windows", "failed_windows"],
        panel_agg_rows,
    )
    with (DATA_DIR / "lobster_panel_summary.txt").open("w") as f:
        f.write("LOBSTER cross-section evidence (1-minute bins, June 21 2012)\n")
        for row in panel_rows:
            f.write(
                f"{row['symbol']} {row['window']}: theta_hat={row['theta_hat']}, "
                f"p={row['p_value']}, ratio={row['deviation_ratio']}, reject={row['reject_5pct']}\n"
            )
        f.write("\naggregate\n")
        for row in panel_agg_rows:
            f.write(
                f"{row['symbol']}: avg_theta_hat={row['avg_theta_hat']}, "
                f"max_deviation_ratio={row['max_deviation_ratio']}, reject_count={row['reject_count']}, "
                f"valid_windows={row['valid_windows']}, failed_windows={row['failed_windows']}\n"
            )

    print("LOBSTER empirical outputs written to", DATA_DIR)


if __name__ == "__main__":
    main()
