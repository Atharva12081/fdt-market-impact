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
DENSE_FREQS = [10 ** (-1.2 + 2.2 * i / 199.0) for i in range(200)]
N_VALUES = [2**12, 2**14, 2**16]
MC_REPS = 500
CRIT_REPS = 1000

BAND_LOW = 0.22
BAND_HIGH = 0.90
EPSILON = 0.18
TAIL_CONSTANT = 0.08
MANIP_SIGNAL_LEVEL = 0.42


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def im_g(omega: float) -> float:
    return sum(a * omega * rho / (rho * rho + omega * omega) for a, rho in MODES)


def s_null(omega: float) -> float:
    return (2.0 * THETA / omega) * im_g(omega)


def stress_multiplier(omega: float) -> float:
    x = math.log10(omega)
    return 1.0 + 1.10 * math.exp(-((x - 0.08) ** 2) / 0.08)


def s_alt(omega: float) -> float:
    return s_null(omega) * stress_multiplier(omega)


def true_delta(omega: float, alternative: bool) -> float:
    spectrum = s_alt(omega) if alternative else s_null(omega)
    return im_g(omega) - omega * spectrum / (2.0 * THETA)


def effective_smoothing(n: int) -> int:
    mapping = {
        2**12: 8,
        2**14: 16,
        2**16: 32,
    }
    return mapping[n]


def averaged_exponential(rng: random.Random, m: int) -> float:
    total = 0.0
    for _ in range(m):
        total += rng.expovariate(1.0)
    return total / m


def simulate_statistic(n: int, rng: random.Random, alternative: bool) -> tuple[float, list[tuple[float, float]]]:
    m = effective_smoothing(n)
    contributions: list[tuple[float, float]] = []
    statistic = 0.0
    for omega in TEST_FREQS:
        true_s = s_alt(omega) if alternative else s_null(omega)
        shat = true_s * averaged_exponential(rng, m)
        dhat = im_g(omega) - omega * shat / (2.0 * THETA)
        variance_proxy = ((omega / (2.0 * THETA)) * true_s) ** 2 / m
        statistic += (dhat * dhat) / variance_proxy
        contributions.append((omega, dhat))
    return statistic, contributions


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[idx]


def trapz(xs: list[float], ys: list[float]) -> float:
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return total


def dense_delta_curve(rng: random.Random, n: int, alternative: bool) -> list[tuple[float, float]]:
    m = effective_smoothing(n)
    curve: list[tuple[float, float]] = []
    for omega in DENSE_FREQS:
        true_s = s_alt(omega) if alternative else s_null(omega)
        shat = true_s * averaged_exponential(rng, m)
        curve.append((omega, im_g(omega) - omega * shat / (2.0 * THETA)))
    return curve


def neighboring_band_metrics(curve: list[tuple[float, float]]) -> tuple[float, float, bool]:
    xs = [omega for omega, _ in curve]
    ds = [delta for _, delta in curve]

    band_indices = [i for i, omega in enumerate(xs) if BAND_LOW <= omega <= BAND_HIGH]
    band_x = [xs[i] for i in band_indices]
    band_d = [ds[i] for i in band_indices]
    if len(band_x) < 2:
        return 0.0, 0.0, False

    positive_band = sorted(max(value, 0.0) for value in band_d)
    delta0_hat = positive_band[max(0, len(positive_band) // 4)]

    signal_kernel = [
        omega * delta0_hat / (((BAND_HIGH + 2.0 * EPSILON) ** 2) - omega * omega)
        for omega in band_x
    ]
    signal_term = (2.0 / math.pi) * trapz(
        band_x,
        signal_kernel,
    )

    outside_x = [omega for omega in xs if omega < BAND_LOW or omega > BAND_HIGH + 2.0 * EPSILON]
    outside_d = [delta for omega, delta in curve if omega < BAND_LOW or omega > BAND_HIGH + 2.0 * EPSILON]
    if len(outside_x) >= 2:
        tail_energy = math.sqrt(trapz(outside_x, [value * value for value in outside_d]))
    else:
        tail_energy = 0.0

    kappa_hat = signal_term - TAIL_CONSTANT * tail_energy / EPSILON

    neighbor_low = BAND_HIGH + EPSILON
    neighbor_high = BAND_HIGH + 2.0 * EPSILON
    neighbor_points = [omega for omega in xs if neighbor_low <= omega <= neighbor_high]
    neighbor_negativity = False
    if neighbor_points:
        real_values: list[float] = []
        for omega in neighbor_points:
            integrand = []
            for omega_prime, delta_prime in curve:
                denom = omega_prime * omega_prime - omega * omega
                if abs(denom) < 1e-12:
                    integrand.append(0.0)
                else:
                    integrand.append(omega_prime * delta_prime / denom)
            real_values.append((2.0 / math.pi) * trapz(xs, integrand))
        neighbor_negativity = all(value < 0.0 for value in real_values)

    return kappa_hat, tail_energy, neighbor_negativity


def manipulation_delta_true(omega: float) -> float:
    center = 0.5 * (BAND_LOW + BAND_HIGH)
    half_width = 0.5 * (BAND_HIGH - BAND_LOW)
    x = abs(omega - center) / half_width
    if x >= 1.0:
        return 0.0
    return MANIP_SIGNAL_LEVEL * (1.0 - x * x)


def simulate_manipulation_statistic(n: int, rng: random.Random, alternative: bool) -> tuple[float, list[tuple[float, float]]]:
    m = effective_smoothing(n)
    noise_sd = 0.18 / math.sqrt(m)
    contributions: list[tuple[float, float]] = []
    statistic = 0.0
    for omega in TEST_FREQS:
        true_delta_value = manipulation_delta_true(omega) if alternative else 0.0
        dhat = true_delta_value + rng.gauss(0.0, noise_sd)
        statistic += (dhat * dhat) / (noise_sd * noise_sd)
        contributions.append((omega, dhat))
    return statistic, contributions


def dense_manipulation_curve(rng: random.Random, n: int, alternative: bool) -> list[tuple[float, float]]:
    m = effective_smoothing(n)
    noise_sd = 0.18 / math.sqrt(m)
    curve: list[tuple[float, float]] = []
    for omega in DENSE_FREQS:
        true_delta_value = manipulation_delta_true(omega) if alternative else 0.0
        curve.append((omega, true_delta_value + rng.gauss(0.0, noise_sd)))
    return curve


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    rng = random.Random(20260404)

    summary_rows: list[dict[str, float]] = []
    example_rows: list[dict[str, float]] = []
    neighbor_rows: list[dict[str, float]] = []
    manipulation_rows: list[dict[str, float]] = []

    for n in N_VALUES:
        crit_draws = [simulate_statistic(n, rng, alternative=False)[0] for _ in range(CRIT_REPS)]
        critical_value = percentile(crit_draws, 0.95)

        null_draws = [simulate_statistic(n, rng, alternative=False)[0] for _ in range(MC_REPS)]
        alt_draws = [simulate_statistic(n, rng, alternative=True)[0] for _ in range(MC_REPS)]
        size = sum(1 for value in null_draws if value > critical_value) / MC_REPS
        power = sum(1 for value in alt_draws if value > critical_value) / MC_REPS

        rejection_count = 0
        criterion_count = 0
        negativity_count = 0
        for _ in range(MC_REPS):
            stat, _ = simulate_statistic(n, rng, alternative=True)
            curve = dense_delta_curve(rng, n, alternative=True)
            if stat > critical_value:
                rejection_count += 1
                kappa_hat, _, neighbor_negativity = neighboring_band_metrics(curve)
                if kappa_hat > 0.0:
                    criterion_count += 1
                if neighbor_negativity:
                    negativity_count += 1

        summary_rows.append(
            {
                "n": n,
                "critical_value": round(critical_value, 4),
                "size": round(size, 3),
                "power": round(power, 3),
                "m": effective_smoothing(n),
            }
        )
        neighbor_rows.append(
            {
                "n": n,
                "rejections": rejection_count,
                "criterion_rate": round(criterion_count / rejection_count, 3) if rejection_count else 0.0,
                "negativity_rate": round(negativity_count / rejection_count, 3) if rejection_count else 0.0,
                "band_low": BAND_LOW,
                "band_high": BAND_HIGH,
                "neighbor_low": BAND_HIGH + EPSILON,
                "neighbor_high": BAND_HIGH + 2.0 * EPSILON,
            }
        )

        manip_crit_draws = [simulate_manipulation_statistic(n, rng, alternative=False)[0] for _ in range(CRIT_REPS)]
        manip_critical_value = percentile(manip_crit_draws, 0.95)
        manip_rejections = 0
        manip_negativity = 0
        for _ in range(MC_REPS):
            stat, _ = simulate_manipulation_statistic(n, rng, alternative=True)
            curve = dense_manipulation_curve(rng, n, alternative=True)
            if stat > manip_critical_value:
                manip_rejections += 1
                _, _, neighbor_negativity = neighboring_band_metrics(curve)
                if neighbor_negativity:
                    manip_negativity += 1
        manipulation_rows.append(
            {
                "n": n,
                "rejections": manip_rejections,
                "negativity_rate": round(manip_negativity / manip_rejections, 3) if manip_rejections else 0.0,
                "critical_value": round(manip_critical_value, 4),
                "band_low": BAND_LOW,
                "band_high": BAND_HIGH,
                "neighbor_low": BAND_HIGH + EPSILON,
                "neighbor_high": BAND_HIGH + 2.0 * EPSILON,
            }
        )

    example_n = 2**14
    m = effective_smoothing(example_n)
    for omega in DENSE_FREQS:
        shat = s_alt(omega) * averaged_exponential(rng, m)
        example_rows.append(
            {
                "omega": omega,
                "true_delta": true_delta(omega, alternative=True),
                "estimated_delta": im_g(omega) - omega * shat / (2.0 * THETA),
            }
        )

    write_csv(
        DATA_DIR / "monte_carlo_results.csv",
        ["n", "critical_value", "size", "power", "m"],
        summary_rows,
    )
    write_csv(
        DATA_DIR / "monte_carlo_delta_example.csv",
        ["omega", "true_delta", "estimated_delta"],
        example_rows,
    )
    write_csv(
        DATA_DIR / "monte_carlo_neighbor_check.csv",
        [
            "n",
            "rejections",
            "criterion_rate",
            "negativity_rate",
            "band_low",
            "band_high",
            "neighbor_low",
            "neighbor_high",
        ],
        neighbor_rows,
    )
    write_csv(
        DATA_DIR / "monte_carlo_manipulation_check.csv",
        [
            "n",
            "rejections",
            "negativity_rate",
            "critical_value",
            "band_low",
            "band_high",
            "neighbor_low",
            "neighbor_high",
        ],
        manipulation_rows,
    )

    with (DATA_DIR / "monte_carlo_results.txt").open("w") as f:
        for row in summary_rows:
            f.write(
                f"n={int(row['n'])}, m={int(row['m'])}, critical={row['critical_value']}, "
                f"size={row['size']}, power={row['power']}\n"
            )
        f.write("\nneighboring-band-check\n")
        for row in neighbor_rows:
            f.write(
                f"n={int(row['n'])}, rejections={int(row['rejections'])}, "
                f"criterion_rate={row['criterion_rate']}, negativity_rate={row['negativity_rate']}, "
                f"band=[{row['band_low']},{row['band_high']}], "
                f"neighbor=[{row['neighbor_low']},{row['neighbor_high']}]\n"
            )
        f.write("\nmanipulation-signal-check\n")
        for row in manipulation_rows:
            f.write(
                f"n={int(row['n'])}, rejections={int(row['rejections'])}, "
                f"negativity_rate={row['negativity_rate']}, critical={row['critical_value']}, "
                f"band=[{row['band_low']},{row['band_high']}], "
                f"neighbor=[{row['neighbor_low']},{row['neighbor_high']}]\n"
            )

    print("Monte Carlo outputs written to", DATA_DIR)


if __name__ == "__main__":
    main()
