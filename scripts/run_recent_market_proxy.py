from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

import run_lobster_empirical as fdt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figures" / "data"

LOOKBACK_PERIOD = "10d"
INTERVAL = "5m"
BOOT_REPS = 300

MARKETS: dict[str, dict[str, object]] = {
    "US": {
        "timezone": "America/New_York",
        "symbols": ["AAPL", "MSFT", "JPM", "XOM", "NVDA"],
    },
    "India": {
        "timezone": "Asia/Kolkata",
        "symbols": ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS"],
    },
}

PROXIES = {
    "signed_volume": lambda ret, price, volume: (1.0 if ret > 0 else -1.0 if ret < 0 else 0.0) * volume,
    "signed_sqrt_volume": lambda ret, price, volume: (1.0 if ret > 0 else -1.0 if ret < 0 else 0.0) * math.sqrt(max(volume, 0.0)),
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_intraday(symbol: str) -> list[dict[str, object]]:
    df = yf.download(symbol, period=LOOKBACK_PERIOD, interval=INTERVAL, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"No intraday data returned for {symbol}.")

    # yfinance may return a single-ticker MultiIndex frame depending on version.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    rows: list[dict[str, object]] = []
    for ts, row in df.iterrows():
        price = float(row["Close"])
        volume = float(row["Volume"])
        if not math.isfinite(price) or not math.isfinite(volume) or price <= 0.0:
            continue
        rows.append({"ts": ts, "close": price, "volume": volume})
    return rows


def split_day_positions(day_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    n = len(day_rows)
    if n < 30:
        return {"Open": [], "Midday": [], "Close": []}
    cut1 = n // 3
    cut2 = (2 * n) // 3
    return {
        "Open": day_rows[:cut1],
        "Midday": day_rows[cut1:cut2],
        "Close": day_rows[cut2:],
    }


def build_phase_series(rows: list[dict[str, object]], timezone_name: str, proxy_name: str) -> tuple[dict[str, list[float]], list[str]]:
    tz = ZoneInfo(timezone_name)
    grouped: dict[str, list[dict[str, object]]] = {}
    dates_seen: list[str] = []
    for row in rows:
        local_ts = row["ts"].astimezone(tz)
        date_key = local_ts.date().isoformat()
        grouped.setdefault(date_key, []).append(row)
    dates_seen = sorted(grouped)

    phase_returns = {"Open": [], "Midday": [], "Close": []}
    phase_flows = {"Open": [], "Midday": [], "Close": []}

    proxy_fn = PROXIES[proxy_name]

    for date_key in dates_seen:
        day_rows = sorted(grouped[date_key], key=lambda x: x["ts"])
        segments = split_day_positions(day_rows)
        for phase, seg_rows in segments.items():
            if len(seg_rows) < 4:
                continue
            prices = [float(x["close"]) for x in seg_rows]
            vols = [float(x["volume"]) for x in seg_rows]
            returns = [math.log(prices[i]) - math.log(prices[i - 1]) for i in range(1, len(prices))]
            flows = [proxy_fn(returns[i - 1], prices[i], vols[i]) for i in range(1, len(prices))]
            phase_returns[phase].extend(returns)
            phase_flows[phase].extend(flows)

    return {phase: [*phase_returns[phase], *phase_flows[phase]] for phase in phase_returns}, dates_seen


def build_phase_returns_flows(rows: list[dict[str, object]], timezone_name: str, proxy_name: str) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    tz = ZoneInfo(timezone_name)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        local_ts = row["ts"].astimezone(tz)
        date_key = local_ts.date().isoformat()
        grouped.setdefault(date_key, []).append(row)
    dates_seen = sorted(grouped)

    proxy_fn = PROXIES[proxy_name]
    out: dict[str, tuple[list[float], list[float]]] = {}

    for phase in ("Open", "Midday", "Close"):
        out[phase] = ([], [])

    for date_key in dates_seen:
        day_rows = sorted(grouped[date_key], key=lambda x: x["ts"])
        segments = split_day_positions(day_rows)
        for phase, seg_rows in segments.items():
            if len(seg_rows) < 4:
                continue
            prices = [float(x["close"]) for x in seg_rows]
            vols = [float(x["volume"]) for x in seg_rows]
            returns = [math.log(prices[i]) - math.log(prices[i - 1]) for i in range(1, len(prices))]
            flows = [proxy_fn(returns[i - 1], prices[i], vols[i]) for i in range(1, len(prices))]
            out[phase][0].extend(returns)
            out[phase][1].extend(flows)

    return out, dates_seen


def analyze_symbol(symbol: str, market: str, timezone_name: str, proxy_name: str, rng: random.Random) -> list[dict[str, object]]:
    rows = fetch_intraday(symbol)
    phase_data, dates_seen = build_phase_returns_flows(rows, timezone_name, proxy_name)
    start_date = dates_seen[0]
    end_date = dates_seen[-1]
    trading_days = len(dates_seen)

    results: list[dict[str, object]] = []
    for phase in ("Open", "Midday", "Close"):
        returns, flows = phase_data[phase]
        if len(returns) < 64:
            results.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "proxy": proxy_name,
                    "phase": phase,
                    "start_date": start_date,
                    "end_date": end_date,
                    "trading_days": trading_days,
                    "n_obs": len(returns),
                    "m": "",
                    "theta_hat": "",
                    "test_stat": "",
                    "critical_value": "",
                    "p_value": "",
                    "deviation_ratio": "",
                    "reject_5pct": "",
                    "error": "Too few observations for spectral analysis.",
                }
            )
            continue
        try:
            summary, _ = fdt.analyze_window(phase, returns, flows, rng)
        except RuntimeError as exc:
            results.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "proxy": proxy_name,
                    "phase": phase,
                    "start_date": start_date,
                    "end_date": end_date,
                    "trading_days": trading_days,
                    "n_obs": len(returns),
                    "m": max(2, int(round(math.sqrt(len(returns)) / 4.0))),
                    "theta_hat": "",
                    "test_stat": "",
                    "critical_value": "",
                    "p_value": "",
                    "deviation_ratio": "",
                    "reject_5pct": "",
                    "error": str(exc),
                }
            )
            continue
        results.append(
            {
                "market": market,
                "symbol": symbol,
                "proxy": proxy_name,
                "phase": phase,
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": trading_days,
                "n_obs": summary["n_obs"],
                "m": summary["m"],
                "theta_hat": summary["theta_hat"],
                "test_stat": summary["test_stat"],
                "critical_value": summary["critical_value"],
                "p_value": summary["p_value"],
                "deviation_ratio": summary["deviation_ratio"],
                "reject_5pct": summary["reject_5pct"],
                "error": "",
            }
        )
    return results


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["market"]), str(row["proxy"])), []).append(row)

    with path.open("w") as f:
        f.write("Recent public-data FDT proxy screen\n")
        f.write("Caveat: order flow is proxied from 5-minute bars, not true signed trade flow.\n\n")
        for (market, proxy), items in sorted(grouped.items()):
            f.write(f"{market} | {proxy}\n")
            by_symbol: dict[str, list[dict[str, object]]] = {}
            for row in items:
                by_symbol.setdefault(str(row["symbol"]), []).append(row)
            for symbol, symbol_rows in sorted(by_symbol.items()):
                valid_rows = [row for row in symbol_rows if not row.get("error")]
                failed_rows = [row for row in symbol_rows if row.get("error")]
                sample_rows = valid_rows or symbol_rows
                sample = f"{sample_rows[0]['start_date']} to {sample_rows[0]['end_date']}"
                if valid_rows:
                    reject_count = sum(int(row["reject_5pct"]) for row in valid_rows)
                    avg_ratio = sum(float(row["deviation_ratio"]) for row in valid_rows) / len(valid_rows)
                    avg_theta = sum(float(row["theta_hat"]) for row in valid_rows) / len(valid_rows)
                    valid_part = (
                        f"valid={len(valid_rows)}/3, rejects={reject_count}/{len(valid_rows)}, "
                        f"avg_theta={avg_theta:.3f}, avg_ratio={avg_ratio:.3f}"
                    )
                else:
                    valid_part = "valid=0/3"
                f.write(
                    f"  {symbol}: {valid_part}, failed={len(failed_rows)}/3, sample={sample}\n"
                )
                for row in valid_rows:
                    f.write(
                        f"    {row['phase']}: theta={float(row['theta_hat']):.3f}, "
                        f"ratio={float(row['deviation_ratio']):.3f}, p={float(row['p_value']):.3f}, "
                        f"reject={int(row['reject_5pct'])}\n"
                    )
                if failed_rows:
                    failed = ", ".join(f"{row['phase']} ({row['error']})" for row in failed_rows)
                    f.write(f"    failed: {failed}\n")
            f.write("\n")


def main() -> None:
    ensure_dirs()
    fdt.BOOT_REPS = BOOT_REPS
    rng = random.Random(20260427)

    all_rows: list[dict[str, object]] = []
    for market, cfg in MARKETS.items():
        timezone_name = str(cfg["timezone"])
        for symbol in cfg["symbols"]:
            try:
                for proxy_name in PROXIES:
                    all_rows.extend(analyze_symbol(symbol, market, timezone_name, proxy_name, rng))
            except Exception as exc:
                all_rows.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "proxy": "error",
                        "phase": "ERROR",
                        "start_date": "",
                        "end_date": "",
                        "trading_days": 0,
                        "n_obs": 0,
                        "m": 0,
                        "theta_hat": "",
                        "test_stat": "",
                        "critical_value": "",
                        "p_value": "",
                        "deviation_ratio": "",
                        "reject_5pct": "",
                        "error": str(exc),
                    }
                )

    fieldnames = [
        "market",
        "symbol",
        "proxy",
        "phase",
        "start_date",
        "end_date",
        "trading_days",
        "n_obs",
        "m",
        "theta_hat",
        "test_stat",
        "critical_value",
        "p_value",
        "deviation_ratio",
        "reject_5pct",
        "error",
    ]
    for row in all_rows:
        row.setdefault("error", "")

    csv_path = DATA_DIR / "recent_market_proxy_results.csv"
    txt_path = DATA_DIR / "recent_market_proxy_summary.txt"
    write_csv(csv_path, fieldnames, all_rows)
    write_summary(txt_path, [row for row in all_rows if row["phase"] != "ERROR"])
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
