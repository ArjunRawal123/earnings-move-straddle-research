"""
collect_snapshots.py — daily ATM straddle collector for EM research.

Captures the options market's implied earnings move for stocks reporting soon.
This data is PERISHABLE: a straddle price the day before earnings exists only
in that moment. Run this every trading day during earnings season.

Usage (from the EM-Research project root):
    python3 src/collect_snapshots.py
    python3 src/collect_snapshots.py --horizon 5
    python3 src/collect_snapshots.py --tickers AAPL,MSFT,TSLA      # testing
    python3 src/collect_snapshots.py --limit 25                    # testing

Output: data/straddle_snapshots/YYYY-MM-DD.csv  (append-only, one row per ticker)

Design notes:
  - Collects WIDE (raw legs, spreads, OI, volume) so liquidity filters can be
    chosen later in analysis rather than baked in irreversibly here.
  - Does NO analysis and NO filtering. It is a dumb, reliable recorder.
  - Failures are logged as rows with status != 'ok' so gaps are visible rather
    than silent (silent gaps become invisible sample bias).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed. Run: pip3 install yfinance")


# ----------------------------------------------------------------------------
# Paths — relative to project root, so run this from EM-Research/
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONSTITUENTS = ROOT / "data" / "constituents.csv"
SNAPSHOT_DIR = ROOT / "data" / "straddle_snapshots"

# Politeness delay between tickers (yfinance is unofficial; don't hammer it)
SLEEP_SECONDS = 0.4


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
def load_universe() -> list[str]:
    """Read S&P 500 constituents. Yahoo uses '-' where the index uses '.'."""
    df = pd.read_csv(CONSTITUENTS)
    symbols = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return sorted(symbols.unique().tolist())


# ----------------------------------------------------------------------------
# Forward earnings calendar
# ----------------------------------------------------------------------------
def next_earnings_date(tk: "yf.Ticker") -> date | None:
    """
    Next scheduled earnings date, or None.

    Tries .calendar first (fast, usually just the next event), then falls back
    to .get_earnings_dates() which includes past events and needs filtering.
    Both are best-effort: Yahoo's calendar is sometimes stale or missing, which
    is exactly why failures get logged rather than swallowed.
    """
    today = date.today()

    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            future = [_to_date(d) for d in dates]
            future = [d for d in future if d and d >= today]
            if future:
                return min(future)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            vals = cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
            future = [_to_date(v) for v in vals]
            future = [d for d in future if d and d >= today]
            if future:
                return min(future)
    except Exception:
        pass

    try:
        df = tk.get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            idx = [_to_date(i) for i in df.index]
            future = [d for d in idx if d and d >= today]
            if future:
                return min(future)
    except Exception:
        pass

    return None


def _to_date(value) -> date | None:
    """Coerce assorted timestamp types to a plain date."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Spot price
# ----------------------------------------------------------------------------
def get_spot(tk: "yf.Ticker") -> float | None:
    try:
        px = tk.fast_info.get("lastPrice")
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    try:
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Straddle measurement
# ----------------------------------------------------------------------------
def pick_expiry(expiries: list[str], earnings_dt: date) -> str | None:
    """
    First expiry STRICTLY AFTER the earnings date.

    Why strictly after: an expiry on or before the announcement prices no event
    risk at all. Because we don't reliably know BMO vs AMC in advance, '> the
    earnings date' covers both cases (a BMO move lands that same day, an AMC
    move the next — both are captured by an expiry after the date itself).
    """
    valid = [e for e in expiries if _to_date(e) and _to_date(e) > earnings_dt]
    return min(valid) if valid else None


def mid(bid, ask, last) -> tuple[float | None, str]:
    """
    Mid price, with source flag.

    Mids over last-traded prices: a 'last' can be hours stale and will quietly
    poison the EM number. Fall back to last only when a side is missing, and
    record which happened so these rows can be excluded in analysis.
    """
    try:
        b, a = float(bid or 0), float(ask or 0)
        if b > 0 and a > 0 and a >= b:
            return (b + a) / 2.0, "mid"
    except Exception:
        pass
    try:
        l = float(last or 0)
        if l > 0:
            return l, "last"
    except Exception:
        pass
    return None, "none"


def snapshot_ticker(ticker: str, horizon_days: int) -> dict:
    """Collect one row. Never raises — failures come back as status rows."""
    now = datetime.now()
    row = {
        "collected_at": now.isoformat(timespec="seconds"),
        "collection_date": now.date().isoformat(),
        "ticker": ticker,
        "status": "ok",
        "note": "",
    }

    try:
        tk = yf.Ticker(ticker)

        earnings_dt = next_earnings_date(tk)
        if earnings_dt is None:
            row.update(status="no_earnings_date", note="calendar unavailable")
            return row

        days_to = (earnings_dt - now.date()).days
        row["earnings_date"] = earnings_dt.isoformat()
        row["days_to_earnings"] = days_to

        if days_to < 0 or days_to > horizon_days:
            row.update(status="outside_horizon")
            return row

        spot = get_spot(tk)
        if not spot:
            row.update(status="no_spot")
            return row
        row["spot"] = round(spot, 4)

        expiries = list(tk.options or [])
        if not expiries:
            row.update(status="no_options")
            return row

        expiry = pick_expiry(expiries, earnings_dt)
        if expiry is None:
            row.update(status="no_expiry_after_earnings")
            return row
        row["expiry"] = expiry
        row["days_expiry_after_earnings"] = (_to_date(expiry) - earnings_dt).days

        chain = tk.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            row.update(status="empty_chain")
            return row

        # ATM = strike closest to spot, and it must exist on BOTH sides
        shared = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
        if not shared:
            row.update(status="no_shared_strike")
            return row
        strike = min(shared, key=lambda k: abs(k - spot))
        row["strike"] = float(strike)
        row["moneyness"] = round(strike / spot - 1.0, 5)

        c = calls.loc[calls["strike"] == strike].iloc[0]
        p = puts.loc[puts["strike"] == strike].iloc[0]

        # Log raw legs so liquidity thresholds are a later decision, not a
        # collection-time one that can never be revisited.
        for leg, s in (("call", c), ("put", p)):
            row[f"{leg}_bid"] = _f(s.get("bid"))
            row[f"{leg}_ask"] = _f(s.get("ask"))
            row[f"{leg}_last"] = _f(s.get("lastPrice"))
            row[f"{leg}_iv"] = _f(s.get("impliedVolatility"))
            row[f"{leg}_volume"] = _f(s.get("volume"))
            row[f"{leg}_oi"] = _f(s.get("openInterest"))

        c_mid, c_src = mid(c.get("bid"), c.get("ask"), c.get("lastPrice"))
        p_mid, p_src = mid(p.get("bid"), p.get("ask"), p.get("lastPrice"))
        row["call_mid"], row["put_mid"] = c_mid, p_mid
        row["price_source"] = c_src if c_src == p_src else f"{c_src}/{p_src}"

        if c_mid is None or p_mid is None:
            row.update(status="no_prices")
            return row

        straddle = c_mid + p_mid
        row["straddle_price"] = round(straddle, 4)
        row["implied_em"] = round(straddle / spot, 6)   # the number this all exists for

        # Relative spread — the raw material for post-hoc liquidity filtering
        row["call_rel_spread"] = _rel_spread(c.get("bid"), c.get("ask"))
        row["put_rel_spread"] = _rel_spread(p.get("bid"), p.get("ask"))

        return row

    except Exception as exc:
        row.update(status="error", note=f"{type(exc).__name__}: {exc}"[:200])
        return row


def _f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _rel_spread(bid, ask):
    try:
        b, a = float(bid or 0), float(ask or 0)
        if b > 0 and a > 0 and a >= b:
            return round((a - b) / ((a + b) / 2.0), 5)
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Daily ATM straddle snapshot collector")
    ap.add_argument("--horizon", type=int, default=5,
                    help="Collect for names reporting within this many days (default 5)")
    ap.add_argument("--tickers", type=str, default=None,
                    help="Comma-separated tickers instead of the full universe (testing)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N tickers (testing)")
    ap.add_argument("--keep-all", action="store_true",
                    help="Write rows for every ticker, including outside-horizon ones")
    args = ap.parse_args()

    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not CONSTITUENTS.exists():
            sys.exit(f"Missing {CONSTITUENTS}")
        universe = load_universe()
    if args.limit:
        universe = universe[: args.limit]

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{date.today().isoformat()}.csv"

    print(f"Scanning {len(universe)} tickers | horizon {args.horizon}d | -> {out_path.name}")
    rows, captured = [], 0

    for i, ticker in enumerate(universe, 1):
        row = snapshot_ticker(ticker, args.horizon)

        if row["status"] == "ok":
            captured += 1
            print(f"  [{i}/{len(universe)}] {ticker:6s} EM={row['implied_em']*100:5.2f}% "
                  f"earnings={row['earnings_date']} (T-{row['days_to_earnings']}) exp={row['expiry']}")
            rows.append(row)
        elif row["status"] == "outside_horizon":
            if args.keep_all:
                rows.append(row)
        else:
            # Log every non-trivial failure: known gaps beat invisible ones.
            print(f"  [{i}/{len(universe)}] {ticker:6s} -- {row['status']} {row['note']}")
            rows.append(row)

        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("Nothing to write.")
        return

    df = pd.DataFrame(rows)
    lead = ["collected_at", "collection_date", "ticker", "status", "earnings_date",
            "days_to_earnings", "spot", "expiry", "strike", "implied_em"]
    ordered = [c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]
    df = df[ordered]

    # Append-only: a rerun adds rows, never destroys the morning's capture.
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)

    print(f"\nCaptured {captured} straddles | {len(rows)} rows appended to {out_path}")
    if captured:
        ok = df[df["status"] == "ok"]
        print(f"Median implied EM: {ok['implied_em'].median()*100:.2f}%")
        print("Reporting soon:", ", ".join(sorted(ok["ticker"].tolist())[:20]))


if __name__ == "__main__":
    main()