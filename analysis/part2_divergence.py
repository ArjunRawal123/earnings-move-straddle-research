"""
part2_divergence.py — Sections 6.2 (divergence) and 6.3 (premium).

Computes realized moves DIRECTLY from prices using the snapshot earnings dates,
rather than joining to event_em.csv (which is built from an EDGAR file that
predates the live-collection season and therefore does not contain these events).

Realized move = |close[reaction] / close[reaction-1] - 1|, where the reaction day
is determined by BMO/AMC timing:
  - AMC (reports after close): reaction = next trading day after earnings_date
  - BMO (reports before open): reaction = earnings_date itself
Timing is taken from data/ticker_timing.csv where available; if a ticker is not
classified, both candidate days are checked and the one with the larger move is
used (with the choice logged), since the market reacts on the true day.

Inputs : data/calibration_events.csv   (implied EM + historical mean EM)
         data/ohlcv_all.csv            (refreshed prices through the season)
         data/ticker_timing.csv        (BMO/AMC per ticker, optional)
Outputs: report/figures/fig8_divergence_buckets.png
         report/tables/part2_divergence.txt
         data/divergence_events.csv

Run (after refresh):  python3 analysis/part2_divergence.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "data" / "calibration_events.csv"
PRICES = ROOT / "data" / "ohlcv_all.csv"
TIMING = ROOT / "data" / "ticker_timing.csv"
FIGDIR = ROOT / "report" / "figures"
TABDIR = ROOT / "report" / "tables"
OUT = ROOT / "data" / "divergence_events.csv"

N_BUCKETS = 5

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
INK, ACCENT, BLUE = "#1f2d3d", "#c0392b", "#2c6e9c"


def load_prices() -> pd.DataFrame:
    px = pd.read_csv(PRICES, parse_dates=["date"])
    px = px.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    return px


def realized_for_event(px_t: pd.DataFrame, earnings_date: pd.Timestamp,
                       timing: str | None):
    """
    Return (abs_move, reaction_date, used_timing) or (None, None, None).

    px_t: this ticker's price rows, sorted by date, with a precomputed 'ret'
          (close/prev_close - 1).
    """
    dates = px_t["date"].values

    def move_on_or_after(anchor):
        i = int(np.searchsorted(dates, np.datetime64(anchor), side="left"))
        if i >= len(px_t):
            return None, None
        row = px_t.iloc[i]
        if pd.isna(row["ret"]):
            return None, None
        return abs(float(row["ret"])), row["date"]

    amc_move, amc_day = move_on_or_after(earnings_date + pd.Timedelta("1D"))
    bmo_move, bmo_day = move_on_or_after(earnings_date)

    if timing == "AMC" and amc_move is not None:
        return amc_move, amc_day, "AMC"
    if timing == "BMO" and bmo_move is not None:
        return bmo_move, bmo_day, "BMO"

    # unknown timing: take the larger of the two candidate days (the market
    # reacts on the true announcement day)
    cands = [(m, d, lab) for m, d, lab in
             [(bmo_move, bmo_day, "BMO?"), (amc_move, amc_day, "AMC?")]
             if m is not None]
    if not cands:
        return None, None, None
    return max(cands, key=lambda x: x[0])


def build(df: pd.DataFrame) -> pd.DataFrame:
    px = load_prices()
    px["ret"] = px.groupby("ticker")["close"].pct_change()

    timing = {}
    if TIMING.exists():
        t = pd.read_csv(TIMING)
        timing = dict(zip(t["ticker"], t["modal_timing"]))

    df = df.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    out = []
    for _, r in df.iterrows():
        px_t = px[px.ticker == r.ticker]
        if px_t.empty:
            continue
        m, day, used = realized_for_event(px_t, r.earnings_date,
                                          timing.get(r.ticker))
        if m is None:
            continue
        out.append({**r.to_dict(), "realized": m,
                    "reaction_date": day, "used_timing": used})
    res = pd.DataFrame(out)
    print(f"Scored {len(res)} of {len(df)} events with realized moves")
    return res


def section_62(df: pd.DataFrame):
    df = df.copy()
    df["r_over_implied"] = df["realized"] / df["implied_em"]
    df["r_over_hist"] = df["realized"] / df["hist_em"]
    df["bucket"] = pd.qcut(df["divergence"], N_BUCKETS,
                           labels=[f"Q{i+1}" for i in range(N_BUCKETS)])
    g = df.groupby("bucket", observed=True).agg(
        n=("divergence", "size"),
        div_median=("divergence", "median"),
        realized_med=("realized", "median"),
        implied_med=("implied_em", "median"),
        hist_med=("hist_em", "median"),
        r_over_implied=("r_over_implied", "median"),
        r_over_hist=("r_over_hist", "median"),
    )
    for c in ["realized_med", "implied_med", "hist_med"]:
        g[c] = (g[c] * 100).round(2)
    g[["div_median", "r_over_implied", "r_over_hist"]] = g[["div_median", "r_over_implied", "r_over_hist"]].round(3)
    return df, g


def make_figure(g: pd.DataFrame):
    x = np.arange(len(g)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - w/2, g["r_over_implied"], w, color=ACCENT, label="realized / implied")
    ax.bar(x + w/2, g["r_over_hist"], w, color=BLUE, label="realized / historical")
    ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n(div {d:.2f})" for i, d in
                        zip(g.index, g["div_median"])])
    ax.set_xlabel("Divergence quintile (implied / historical, low \u2192 high)")
    ax.set_ylabel("Realized \u00f7 forecast (median)")
    ax.set_title("When implied diverges from history, which forecast wins?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_divergence_buckets.png")
    plt.close(fig)


def section_63(df: pd.DataFrame) -> dict:
    r_imp = df["realized"] / df["implied_em"]
    r_hist = df["realized"] / df["hist_em"]
    return {
        "n": len(df),
        "median_r_over_implied": r_imp.median(),
        "mean_r_over_implied": r_imp.mean(),
        "median_r_over_hist": r_hist.median(),
        "share_realized_below_implied": (df["realized"] < df["implied_em"]).mean(),
        "median_implied": df["implied_em"].median()*100,
        "median_realized": df["realized"].median()*100,
    }


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)
    if not CAL.exists():
        raise SystemExit("Run part2_calibration.py first.")

    cal = pd.read_csv(CAL)
    df = build(cal)
    if len(df) < 20:
        raise SystemExit(f"Only {len(df)} scored — check prices/dates.")
    df.to_csv(OUT, index=False)

    df, buckets = section_62(df)
    make_figure(buckets)
    prem = section_63(df)

    lines = []
    def P(s=""):
        lines.append(s); print(s)

    P("=" * 66)
    P("SECTION 6.2 — DIVERGENCE: WHICH FORECAST DOES REALIZED VINDICATE?")
    P("=" * 66)
    P(buckets.to_string())
    P("")
    P("In each quintile, a column near 1.00 means that forecast matched realized.")
    P("Focus on Q5 (largest implied>>historical divergence):")
    P("  r_over_implied ~1  -> market was right to diverge (saw real info)")
    P("  r_over_hist ~1      -> history was right, market overpriced the move")
    P("")
    P("=" * 66)
    P("SECTION 6.3 — THE PREMIUM")
    P("=" * 66)
    P(f"Events:                        {prem['n']}")
    P(f"Median realized / implied:     {prem['median_r_over_implied']:.3f}")
    P(f"Mean realized / implied:       {prem['mean_r_over_implied']:.3f}")
    P(f"Median realized / historical:  {prem['median_r_over_hist']:.3f}")
    P(f"Share realized < implied:      {prem['share_realized_below_implied']*100:.1f}%")
    P(f"Median implied EM:             {prem['median_implied']:.2f}%")
    P(f"Median realized move:          {prem['median_realized']:.2f}%")
    P("")
    roi = prem['median_r_over_implied']
    if roi < 1:
        P(f"  realized/implied = {roi:.2f} < 1: straddle priced MORE than delivered")
        P(f"  -> earnings volatility risk premium ~{(1-roi)*100:.0f}% at the median.")
    else:
        P(f"  realized/implied = {roi:.2f} >= 1: no systematic overpricing here.")
    P("=" * 66)
    P("Figure -> report/figures/fig8_divergence_buckets.png")
    P(f"Per-event table -> {OUT}")
    (TABDIR / "part2_divergence.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()