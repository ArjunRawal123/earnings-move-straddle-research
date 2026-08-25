"""
realized_em.py — per-event realized earnings moves + trailing historical EM.

Builds columns 1 and 3 of the study's three-column table, entirely from data
already on disk. No network calls, no options data.

    column 1  hist_em_{4,8,12}q   trailing median of PRIOR absolute moves
    column 3  abs_move            the move actually realized on this event

Input : data/earnings_events.csv
        data/ohlcv_all.csv
Output: data/event_em.csv         (one row per kept event)

Definition of the realized move — a one-day close-to-close return on the
reaction day:

    abs_move = | close[t] / close[t-1] - 1 |,  t = reaction_date

`effective_date` in the events file already carries the BMO/AMC shift (AMC
events moved forward one day), so this single formula is correct for all three
timing cases. For an AMC report the prior close is the last print before the
announcement; for a BMO report it's the previous session's close. Either way
the move is measured from the last unaffected close through the first full
session of reaction.

One day rather than multi-day: this isolates the announcement response from
post-earnings-announcement drift, which is a separate phenomenon. A 2-day
variant is included as a robustness column, not as the headline.

Three data quirks are handled here rather than by editing the raw files:

  1. Weekend effective dates. The EDGAR fetcher incremented AMC events by one
     CALENDAR day, so Friday-AMC events landed on Saturday (253 rows). Every
     reaction date is snapped forward to the next real trading date present in
     the price file, which also absorbs market holidays.

  2. Duplicate operational 8-Ks. Some tickers file a second 8-K per quarter
     that isn't an earnings report — TSLA's quarterly delivery release is the
     clearest case (88 events vs an expected ~44, filed on the 2nd of each
     quarter-opening month). Events clustered within 30 days are collapsed,
     preferring the AMC/BMO-timed filing over a mid-session one and, failing
     that, the later filing.

  3. Look-ahead. Trailing medians use only events STRICTLY BEFORE the current
     one (shift(1) before rolling), and require a full window.

Usage (from the EM-Research project root):
    python3 src/realized_em.py
    python3 src/realized_em.py --no-dedup      # keep every 8-K
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TIMING_CHECK = ROOT / "data" / "timing_price_check.csv"
OVERRIDE_RATIO = 1.25
EVENTS = ROOT / "data" / "earnings_events.csv"
PRICES = ROOT / "data" / "ohlcv_all.csv"
OUT = ROOT / "data" / "event_em.csv"

WINDOWS = [4, 8, 12]        # trailing quarters; let the data pick the winner
DEDUP_DAYS = 30             # events closer than this are the same quarter
AMBIG_HOUR = 14             # acceptance >= 14:00 -> timing flag


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    """
    Load OHLCV and normalise to columns: ticker, date, close.

    Column names vary between pipelines, so detect rather than assume. Prefers
    an adjusted close where one exists — using raw closes would turn every
    stock split into a ~50% "earnings move".
    """
    px = pd.read_csv(PRICES)
    cols = {c.lower().strip(): c for c in px.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_ticker = pick("ticker", "symbol", "sym")
    c_date = pick("date", "datetime", "timestamp")
    c_close = pick("adj_close", "adj close", "adjclose", "adjusted_close")
    used_adj = c_close is not None
    if c_close is None:
        c_close = pick("close", "close_adj", "px_close")

    missing = [n for n, c in [("ticker", c_ticker), ("date", c_date),
                              ("close", c_close)] if c is None]
    if missing:
        raise SystemExit(
            f"Could not find column(s) {missing} in {PRICES.name}. "
            f"Columns present: {list(px.columns)}"
        )

    px = px[[c_ticker, c_date, c_close]].copy()
    px.columns = ["ticker", "date", "close"]
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["ticker", "date", "close"])
    px = px.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])

    print(f"Prices: {len(px):,} rows | {px.ticker.nunique()} tickers | "
          f"{px.date.min().date()} -> {px.date.max().date()} | "
          f"{'ADJUSTED' if used_adj else 'RAW closes (watch for splits)'}")
    return px


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS)
    ev["filing_date"] = pd.to_datetime(ev["filing_date"], errors="coerce")
    ev["effective_date"] = pd.to_datetime(ev["effective_date"], errors="coerce")
    ts = pd.to_datetime(ev["acceptance_dt_et"], errors="coerce")
    ev["acc_hour"] = ts.dt.hour
    # Late-afternoon acceptances: can't tell from the timestamp whether the
    # reaction landed same-day or next. Flagged, kept, and re-tested later.
    ev["ambiguous_timing"] = ev["acc_hour"] >= AMBIG_HOUR
    ev = ev.dropna(subset=["filing_date", "effective_date"])
    print(f"Events: {len(ev):,} rows | {ev.ticker.nunique()} tickers")
    return ev


def dedup_events(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse non-earnings 8-Ks filed near a real earnings report.

    Within each ticker, events within DEDUP_DAYS of one another form a cluster;
    one survives. Preference order inside a cluster:
        1. a filing with explicit BMO/AMC timing over a mid-session one
           (operational releases tend to hit mid-session; earnings cluster
            around the open and close)
        2. the later filing (earnings usually follows an operational release)
    """
    ev = ev.sort_values(["ticker", "filing_date"]).copy()
    ev["cluster"] = (
        ev.groupby("ticker")["filing_date"]
          .diff().dt.days.gt(DEDUP_DAYS)
          .fillna(True).cumsum()
    )
    ev["_pref"] = np.where(ev["timing"].isin(["AMC", "BMO"]), 1, 0)
    ev = ev.sort_values(["ticker", "cluster", "_pref", "filing_date"])
    kept = ev.groupby(["ticker", "cluster"], as_index=False).tail(1)
    dropped = len(ev) - len(kept)
    print(f"De-dup: dropped {dropped:,} clustered 8-Ks "
          f"({dropped/len(ev)*100:.1f}%), {len(kept):,} events remain")
    return kept.drop(columns=["_pref"]).sort_values(["ticker", "filing_date"])

# ============================ PASTE FROM HERE ============================
 
def load_timing_overrides(events: "pd.DataFrame") -> dict:
    """
    Return {ticker: 'BMO'|'AMC'} for tickers whose price-identified reaction day
    disagrees with the stored timing by a decisive margin.
 
    Reads data/timing_price_check.csv (produced by the diagnostic one-liner).
    If the file is absent, returns {} and the pipeline behaves exactly as before,
    so this is a safe, optional layer.
    """
    import pandas as pd
    from pathlib import Path
 
    path = TIMING_CHECK
    if not Path(path).exists():
        print("No timing_price_check.csv found — skipping timing overrides.")
        return {}
 
    r = pd.read_csv(path)
    r["ratio"] = r[["filing_day", "next_day"]].max(axis=1) / \
                 r[["filing_day", "next_day"]].min(axis=1)
    strong = r[r["mismatch"] & (r["ratio"] >= OVERRIDE_RATIO)]
    overrides = dict(zip(strong["ticker"], strong["implied_by_price"]))
    print(f"Timing overrides applied to {len(overrides)} tickers: "
          f"{sorted(overrides)}")
    return overrides
 
 
def apply_timing_overrides(ev: "pd.DataFrame", overrides: dict) -> "pd.DataFrame":
    """
    Recompute effective_date for overridden tickers from the corrected timing.
 
        BMO  -> reaction is the filing day     (effective_date = filing_date)
        AMC  -> reaction is the next day        (effective_date = filing_date + 1 cal day;
                                                 compute_moves snaps it to a real
                                                 trading day, so a calendar +1 is fine)
 
    Only the effective_date is changed; the raw `timing` column is left intact so
    the correction is auditable (you can see stored-vs-used disagree).
    """
    import pandas as pd
 
    ev = ev.copy()
    ev["timing_corrected"] = ev["timing"]
    for tkr, corrected in overrides.items():
        mask = ev["ticker"] == tkr
        ev.loc[mask, "timing_corrected"] = corrected
        if corrected == "BMO":
            ev.loc[mask, "effective_date"] = ev.loc[mask, "filing_date"]
        else:  # AMC
            ev.loc[mask, "effective_date"] = ev.loc[mask, "filing_date"] + pd.Timedelta("1D")
    return ev
 
# ============================ PASTE TO HERE ============================
 
# ---------------------------------------------------------------------------
# Move computation
# ---------------------------------------------------------------------------
def compute_moves(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, snap the effective date forward to a real trading day and
    take the close-to-close return on that day (plus a 2-day variant).
    """
    px = px.copy()
    px["ret1"] = px.groupby("ticker")["close"].pct_change()
    px["prev_close"] = px.groupby("ticker")["close"].shift(1)
    px["close_p1"] = px.groupby("ticker")["close"].shift(-1)

    out = []
    for ticker, g_ev in ev.groupby("ticker"):
        g_px = px[px.ticker == ticker]
        if g_px.empty:
            for _, r in g_ev.iterrows():
                out.append({**r.to_dict(), "status": "no_price_data"})
            continue

        dates = g_px["date"].values
        for _, r in g_ev.iterrows():
            eff = np.datetime64(r["effective_date"])
            # Quirk 1: snap to the next available trading date. Fixes the 253
            # Saturday effective dates and any holiday collisions at once.
            i = int(np.searchsorted(dates, eff, side="left"))
            if i >= len(dates):
                out.append({**r.to_dict(), "status": "after_price_history"})
                continue

            row = g_px.iloc[i]
            snapped = int((row["date"] - r["effective_date"]).days)
            if snapped > 5 or pd.isna(row["ret1"]):
                out.append({**r.to_dict(), "status": "no_clean_move"})
                continue

            ret1 = float(row["ret1"])
            ret2 = (float(row["close_p1"] / row["prev_close"] - 1.0)
                    if pd.notna(row.get("close_p1")) and pd.notna(row.get("prev_close"))
                    else np.nan)

            out.append({
                **r.to_dict(),
                "status": "ok",
                "reaction_date": row["date"],
                "days_snapped": snapped,
                "prev_close": float(row["prev_close"]),
                "close": float(row["close"]),
                "signed_move": ret1,
                "abs_move": abs(ret1),
                "abs_move_2d": abs(ret2) if pd.notna(ret2) else np.nan,
            })

    df = pd.DataFrame(out)
    print("Move computation:", df["status"].value_counts().to_dict())
    return df


def add_trailing_em(df):
    ok = df[df["status"] == "ok"].copy()
    ok = ok.sort_values(["ticker", "reaction_date"]).reset_index(drop=True)
    for w in WINDOWS:
        ok[f"hist_em_{w}q"] = (
            ok.groupby("ticker")["abs_move"]
              .transform(lambda s: s.shift(1).rolling(w, min_periods=w).median())
        )
        ok[f"hist_em_mean_{w}q"] = (
            ok.groupby("ticker")["abs_move"]
              .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        )
    ok["n_prior"] = ok.groupby("ticker").cumcount()
    for w in WINDOWS:
        ok[f"em_ratio_{w}q"] = ok["abs_move"] / ok[f"hist_em_{w}q"]
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep every 8-K, including operational releases")
    args = ap.parse_args()

    for p in (EVENTS, PRICES):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    px = load_prices()
    ev = load_events()
    if not args.no_dedup:
        ev = dedup_events(ev)
    overrides = load_timing_overrides(ev)
    ev = apply_timing_overrides(ev, overrides)
    df = compute_moves(ev, px)
    out = add_trailing_em(df)

    keep = ["ticker", "filing_date", "timing", "acc_hour", "ambiguous_timing",
            "effective_date", "reaction_date", "days_snapped",
            "prev_close", "close", "signed_move", "abs_move", "abs_move_2d",
            "n_prior"] + [f"hist_em_{w}q" for w in WINDOWS] \
                      + [f"hist_em_mean_{w}q" for w in WINDOWS] \
                      + [f"em_ratio_{w}q" for w in WINDOWS]
    
    out[[c for c in keep if c in out.columns]].to_csv(OUT, index=False)
    print(f"\nWrote {len(out):,} scored events -> {OUT}")

    # ---- diagnostics ------------------------------------------------------
    print(f"\nMedian |move|: {out.abs_move.median()*100:.2f}%   "
          f"mean: {out.abs_move.mean()*100:.2f}%   "
          f"p95: {out.abs_move.quantile(0.95)*100:.2f}%")
    print(f"Snapped dates: {(out.days_snapped>0).sum():,} events moved forward "
          f"to a trading day")

    for w in WINDOWS:
        sub = out.dropna(subset=[f"hist_em_{w}q"])
        if len(sub) < 100:
            continue
        # Persistence: how well does the trailing median predict the NEXT move?
        # Spearman because both series are heavily right-skewed.
        rho = sub["abs_move"].corr(sub[f"hist_em_{w}q"], method="spearman")
        med_ratio = (sub["abs_move"] / sub[f"hist_em_{w}q"]).median()
        print(f"  {w:2d}q window: n={len(sub):,}  spearman={rho:.3f}  "
              f"median realized/hist={med_ratio:.3f}")

    big = out.nlargest(8, "abs_move")[["ticker", "reaction_date", "abs_move"]]
    print("\nLargest moves (sanity-check these are real, not splits):")
    print(big.assign(abs_move=lambda d: (d.abs_move*100).round(1))
             .to_string(index=False))


if __name__ == "__main__":
    main()