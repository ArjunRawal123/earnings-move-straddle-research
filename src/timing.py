"""
timing.py — per-ticker reporting-habit classifier.

Answers one question: does this company habitually report BMO or AMC?

Needed because the forward calendar gives a DATE but not a TIME. Without this,
a T-0 snapshot is ambiguous: for an AMC reporter it's the ideal measurement
(hours before the release); for a BMO reporter it's taken AFTER the stock has
already moved, and the event vol is already crushed out of the options. Those
rows must be dropped, not merely deprioritized.

Input : data/earnings_events.csv   (offline, no network calls)
Output: data/ticker_timing.csv     (one row per ticker)

Classification rule (ported from Compass `infer_timing`, with one correction):
    acceptance < 09:30            -> BMO
    09:30 <= acceptance < 14:00   -> BMO   [see note]
    14:00 <= acceptance < 16:00   -> ambiguous, excluded from the vote
    acceptance >= 16:00           -> AMC

Note on the mid-morning band: EDGAR acceptance is the 8-K FILING time, not the
announcement time. Companies that report before the open typically issue the
press release at 6-7am but don't file until mid-morning. The acceptance-hour
histogram confirms this: 86% of "INTRADAY" events land 10:00-13:59, with none
before 10:00. Those are before-open reports, so they collapse to BMO. This
matches the effective_date convention already used by the EDGAR fetcher
(BMO/INTRADAY -> same day), so dating and classification agree by construction.

Unlike the Compass runtime version, this does NOT require unanimity. Returning
null on mixed history would discard exactly the names that need a rule. Instead
it reports the modal label plus an agreement share, so the analysis can trust
AAPL (43/43 AMC) blindly and flag a 60/40 name for individual handling.

Usage (from the EM-Research project root):
    python3 src/timing.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "earnings_events.csv"
OUT = ROOT / "data" / "ticker_timing.csv"

# Hour boundaries (America/New_York, as stored in acceptance_dt_et)
BMO_CUTOFF = 14          # accepted before 14:00 -> treated as before-open
AMC_CUTOFF = 16          # accepted 16:00 or later -> after-close
RECENT_N = 8             # events used for the regime-switch check


def classify_row(hour: float | None) -> str:
    """Map an acceptance hour to BMO / AMC / AMBIGUOUS."""
    if hour is None or pd.isna(hour):
        return "UNKNOWN"
    if hour < BMO_CUTOFF:
        return "BMO"
    if hour >= AMC_CUTOFF:
        return "AMC"
    return "AMBIGUOUS"          # 14:00-15:59: could be either, don't guess


def build() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS)

    # Parse the acceptance hour from the raw timestamp rather than trusting the
    # stored `timing` column, which mislabels most before-open reports.
    ts = pd.to_datetime(ev["acceptance_dt_et"], errors="coerce")
    ev["acc_hour"] = ts.dt.hour
    ev["inferred"] = ev["acc_hour"].apply(classify_row)
    ev["event_date"] = pd.to_datetime(ev["filing_date"], errors="coerce")

    print("Inferred timing across all events:")
    for k, v in ev["inferred"].value_counts().items():
        print(f"  {k:10s} {v:6d}  ({v/len(ev)*100:.1f}%)")

    votes = ev[ev["inferred"].isin(["BMO", "AMC"])].copy()
    rows = []

    for ticker, g in votes.groupby("ticker"):
        g = g.sort_values("event_date")
        n_bmo = int((g["inferred"] == "BMO").sum())
        n_amc = int((g["inferred"] == "AMC").sum())
        n = n_bmo + n_amc
        if n == 0:
            continue

        modal = "BMO" if n_bmo >= n_amc else "AMC"
        agreement = max(n_bmo, n_amc) / n

        # Regime check: companies do switch habits. A decade-wide mode would
        # hide a recent change, so compare against the last few events.
        recent = g.tail(RECENT_N)["inferred"]
        r_bmo = int((recent == "BMO").sum())
        r_amc = int((recent == "AMC").sum())
        recent_modal = "BMO" if r_bmo >= r_amc else "AMC"

        if agreement >= 0.90 and n >= 8:
            conf = "high"
        elif agreement >= 0.70 and n >= 4:
            conf = "medium"
        else:
            conf = "low"

        rows.append({
            "ticker": ticker,
            "modal_timing": modal,
            "agreement_share": round(agreement, 3),
            "n_events": n,
            "n_bmo": n_bmo,
            "n_amc": n_amc,
            "recent_timing": recent_modal,
            "switched": recent_modal != modal,
            "confidence": conf,
        })

    out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    # Cross-check the dating convention in the events file. An AMC event whose
    # effective_date equals its filing_date was never shifted forward, so its
    # realized move would be measured on the wrong session.
    chk = ev.merge(out[["ticker", "modal_timing"]], on="ticker", how="left")
    amc = chk[(chk["modal_timing"] == "AMC") & (chk["inferred"] == "AMC")]
    unshifted = (amc["effective_date"] == amc["filing_date"]).sum()
    print(f"\nDating cross-check: {unshifted} AMC events not shifted forward "
          f"(expect ~0)")

    return out


def main():
    if not EVENTS.exists():
        raise SystemExit(f"Missing {EVENTS}")

    out = build()
    out.to_csv(OUT, index=False)

    print(f"\nClassified {len(out)} tickers -> {OUT}")
    print(f"  BMO: {(out.modal_timing=='BMO').sum()}   "
          f"AMC: {(out.modal_timing=='AMC').sum()}")
    print("  confidence:", out["confidence"].value_counts().to_dict())

    switched = out[out["switched"]]
    print(f"\n{len(switched)} tickers appear to have switched timing regime:")
    if len(switched):
        print(switched[["ticker", "modal_timing", "recent_timing",
                        "agreement_share", "n_events"]].head(15).to_string(index=False))

    low = out[out["confidence"] == "low"]
    print(f"\n{len(low)} low-confidence tickers (handle individually):")
    if len(low):
        print(low[["ticker", "n_bmo", "n_amc", "agreement_share"]]
              .head(15).to_string(index=False))

    # Spot-checks against known behaviour.
    for t in ["AAPL", "JPM", "MSFT", "VZ", "MMM", "SCHW"]:
        r = out[out.ticker == t]
        if len(r):
            r = r.iloc[0]
            print(f"  {t:6s} {r.modal_timing}  agreement={r.agreement_share}  n={r.n_events}")


if __name__ == "__main__":
    main()