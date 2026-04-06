from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figures" / "data"

THETA = 0.35
MODES = [(1.2, 0.4), (0.7, 2.2)]
TEST_FREQS = [0.12, 0.28, 0.65, 1.5, 3.2]
DENSE_FREQS = [10 ** (-1.2 + 2.2 * i / 39.0) for i in range(40)]
SESSION_COUNT = 60
SESSION_N = 2**14
SESSION_M = 16
BOOT_CRITICAL = 11.6907


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def im_g(omega: float) -> float:
    return sum(a * omega * rho / (rho * rho + omega * omega) for a, rho in MODES)


def s_null(omega: float) -> float:
    return (2.0 * THETA / omega) * im_g(omega)


def regime_name(session: int) -> str:
    if session <= 20:
        return "equilibrium"
    if session <= 40:
        return "transition"
    return "stress"


def regime_multiplier(session: int, omega: float) -> float:
    x = math.log10(omega)
    bell = math.exp(-((x - 0.10) ** 2) / 0.075)
    if session <= 20:
        return 1.0
    if session <= 40:
        ramp = (session - 20) / 20.0
        return 1.0 + 0.80 * ramp * bell
    ramp = min(1.0, 0.55 + (session - 40) / 20.0)
    shoulder = math.exp(-((x + 0.35) ** 2) / 0.15)
    return 1.0 + 3.00 * ramp * bell + 0.60 * ramp * shoulder


def averaged_exponential(rng: random.Random, m: int) -> float:
    total = 0.0
    for _ in range(m):
        total += rng.expovariate(1.0)
    return total / m


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    rng = random.Random(20260405)

    session_rows: list[dict[str, float | int | str]] = []
    heatmap_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    grouped: dict[str, list[dict[str, float | int | str]]] = {
        "equilibrium": [],
        "transition": [],
        "stress": [],
    }

    for session in range(1, SESSION_COUNT + 1):
        regime = regime_name(session)
        test_stat = 0.0
        theta_ratio_sum = 0.0
        peak_abs_delta = 0.0

        for omega in TEST_FREQS:
            mult = regime_multiplier(session, omega)
            shat = s_null(omega) * mult * averaged_exponential(rng, SESSION_M)
            delta_hat = im_g(omega) - omega * shat / (2.0 * THETA)
            variance_proxy = ((omega / (2.0 * THETA)) * s_null(omega) * mult) ** 2 / SESSION_M
            test_stat += (delta_hat * delta_hat) / variance_proxy
            theta_ratio_sum += omega * shat / (2.0 * im_g(omega))

        for omega in DENSE_FREQS:
            mult = regime_multiplier(session, omega)
            shat = s_null(omega) * mult * averaged_exponential(rng, SESSION_M)
            delta_hat = im_g(omega) - omega * shat / (2.0 * THETA)
            peak_abs_delta = max(peak_abs_delta, abs(delta_hat))
            heatmap_rows.append(
                {
                    "session": session,
                    "log10_omega": round(math.log10(omega), 6),
                    "delta_hat": round(delta_hat, 6),
                    "regime_code": 0 if regime == "equilibrium" else 1 if regime == "transition" else 2,
                }
            )

        theta_hat = theta_ratio_sum / len(TEST_FREQS)
        reject = int(test_stat > BOOT_CRITICAL)
        row = {
            "session": session,
            "regime": regime,
            "theta_hat": round(theta_hat, 5),
            "test_stat": round(test_stat, 4),
            "critical_value": BOOT_CRITICAL,
            "reject": reject,
            "peak_abs_delta": round(peak_abs_delta, 5),
        }
        session_rows.append(row)
        grouped[regime].append(row)

    for regime, rows in grouped.items():
        rejection_rate = sum(int(row["reject"]) for row in rows) / len(rows)
        avg_theta = sum(float(row["theta_hat"]) for row in rows) / len(rows)
        avg_peak = sum(float(row["peak_abs_delta"]) for row in rows) / len(rows)
        summary_rows.append(
            {
                "regime": regime,
                "sessions": len(rows),
                "rejection_rate": round(rejection_rate, 3),
                "avg_theta_hat": round(avg_theta, 5),
                "avg_peak_abs_delta": round(avg_peak, 5),
            }
        )

    write_csv(
        DATA_DIR / "operational_validation_sessions.csv",
        ["session", "regime", "theta_hat", "test_stat", "critical_value", "reject", "peak_abs_delta"],
        session_rows,
    )
    write_csv(
        DATA_DIR / "operational_validation_heatmap.csv",
        ["session", "log10_omega", "delta_hat", "regime_code"],
        heatmap_rows,
    )
    write_csv(
        DATA_DIR / "operational_validation_summary.csv",
        ["regime", "sessions", "rejection_rate", "avg_theta_hat", "avg_peak_abs_delta"],
        summary_rows,
    )

    with (DATA_DIR / "operational_validation_summary.txt").open("w") as f:
        f.write("Rolling-window operational validation\n")
        for row in summary_rows:
            f.write(
                f"{row['regime']}: sessions={row['sessions']}, rejection_rate={row['rejection_rate']}, "
                f"avg_theta_hat={row['avg_theta_hat']}, avg_peak_abs_delta={row['avg_peak_abs_delta']}\n"
            )

    print("Operational validation outputs written to", DATA_DIR)


if __name__ == "__main__":
    main()
