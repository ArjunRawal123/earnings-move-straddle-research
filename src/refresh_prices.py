"""
refresh_prices.py — re-pull OHLCV for the full universe, adjusted closes.

Replaces data/ohlcv_all.csv with fresh data through today, using ADJUSTED
closes (auto_adjust=True). This retires the raw-close split-risk limitation:
every price is split/dividend-adjusted, so a split can no longer masquerade as
an earnings move.

Backs up the existing file first. Pulls in batches with a politeness delay.

Run from the project root:
    python3 src/refresh_prices.py
    python3 src/refresh_prices.py --start 2016-01-01 --end 2026-08-08
"""

from __future__ import annotations
import argparse
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("pip3 install yfinance")

ROOT = Path(__file__).resolve().parent.parent
CONSTITUENTS = ROOT / "data" / "constituents.csv"
OUT = ROOT / "data" / "ohlcv_all.csv"

BATCH = 40
SLEEP = 1.0


def load_universe() -> list[str]:
    df = pd.read_csv(CONSTITUENTS)
    return sorted(df["Symbol"].astype(str).str.strip()
                  .str.replace(".", "-", regex=False).unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat(),
                    help="exclusive end date; default tomorrow to include today's close")
    args = ap.parse_args()

    tickers = load_universe()
    print(f"Refreshing {len(tickers)} tickers | {args.start} -> {args.end} | ADJUSTED")

    # back up the existing file
    if OUT.exists():
        bak = OUT.with_suffix(".csv.pre_refresh_bak")
        shutil.copy(OUT, bak)
        print(f"Backed up existing file -> {bak.name}")

    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        print(f"  batch {i//BATCH + 1}: {batch[0]}..{batch[-1]}")
        try:
            data = yf.download(batch, start=args.start, end=args.end,
                               auto_adjust=True, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            print(f"    batch failed: {e}")
            continue

        for t in batch:
            try:
                sub = data[t] if len(batch) > 1 else data
                sub = sub.dropna(subset=["Close"]).reset_index()
                if sub.empty:
                    continue
                out = pd.DataFrame({
                    "ticker": t,
                    "date": pd.to_datetime(sub["Date"]).dt.date,
                    "open": sub["Open"].values,
                    "high": sub["High"].values,
                    "low": sub["Low"].values,
                    "close": sub["Close"].values,
                    "volume": sub["Volume"].values,
                })
                frames.append(out)
            except Exception:
                continue
        time.sleep(SLEEP)

    if not frames:
        raise SystemExit("No data pulled — aborting, original file preserved via backup.")

    allpx = pd.concat(frames, ignore_index=True)
    allpx = allpx.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    allpx.to_csv(OUT, index=False)

    print(f"\nWrote {len(allpx):,} rows | {allpx.ticker.nunique()} tickers "
          f"| {allpx.date.min()} -> {allpx.date.max()}")
    print("Columns:", list(allpx.columns), "(adjusted closes)")
    print("\nNext: rerun  python3 src/realized_em.py  to score the new events.")


if __name__ == "__main__":
    main()