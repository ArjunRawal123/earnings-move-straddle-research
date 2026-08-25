"""
part2_calibration.py — Section 6.1 (Calibration): does implied EM anchor to history?

Compares the option-implied expected move against the stock's trailing historical
mean EM across the clean primary sample. NO realized moves needed — this section
compares two FORECASTS (the market's vs history's), so it runs before the price
refresh.

Benchmark is the trailing MEAN historical EM (hist_em_mean_8q), not the median:
the straddle prices a mean absolute move, so the mean is the like-for-like
comparison. Using the median would let fat-tailed names sit artificially above
the line (the mean/median wedge) and manufacture false divergences.

Inputs : data/straddle_snapshots/*.csv   (implied EM, collected daily)
         data/event_em.csv               (historical EM, from realized_em.py)
         data/constituents.csv           (GICS sectors)
Outputs: report/figures/fig6_calibration.png
         report/figures/fig7_calibration_by_sector.png
         report/tables/part2_calibration.txt
         data/calibration_events.csv      (the per-event table, for Section 6.2)

Run from the project root:
    python3 analysis/part2_calibration.py
"""

from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "data" / "straddle_snapshots"
EVENT_EM = ROOT / "data" / "event_em.csv"
CONSTITUENTS = ROOT / "data" / "constituents.csv"
FIGDIR = ROOT / "report" / "figures"
TABDIR = ROOT / "report" / "tables"
OUT_TABLE = ROOT / "data" / "calibration_events.csv"

MAX_EXPIRY_GAP = 3          # clean sample: expiry within 3 days of earnings

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
INK, ACCENT = "#1f2d3d", "#c0392b"


# ---------------------------------------------------------------------------
def load_snapshots() -> pd.DataFrame:
    """All ok snapshots, deduped to the latest per (ticker, collection_date)."""
    files = sorted(glob.glob(str(SNAP_DIR / "*.csv")))
    if not files:
        raise SystemExit("No snapshot files found.")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = d[d["status"] == "ok"].copy()
    ok = ok.drop_duplicates(["ticker", "collection_date"], keep="last")
    print(f"Loaded {len(files)} snapshot files | {len(ok)} ok snapshots")
    return ok


def primary_events(ok: pd.DataFrame) -> pd.DataFrame:
    """
    One row per event: the T-1/T-0 pre-announcement snapshot.

    If both T-1 and T-0 exist for an event, keep the one closest to the event
    (smallest non-negative days_to_earnings) — the cleanest pre-announcement read.
    """
    pre = ok[ok["days_to_earnings"].between(0, 1)].copy()
    pre = pre.sort_values("days_to_earnings")  # 0 before 1
    pre = pre.drop_duplicates(["ticker", "earnings_date"], keep="first")
    print(f"Unique events with T-1/T-0 snapshot: {len(pre)}")
    return pre


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the clean-sample screens: expiry gap, quote source, parity."""
    df = df.copy()

    # parity gap for stale-spot detection
    def parity(r):
        try:
            implied_spot = r["strike"] + (r["call_bid"] + r["call_ask"]) / 2 \
                                        - (r["put_bid"] + r["put_ask"]) / 2
            return abs(implied_spot - r["spot"]) / r["spot"]
        except Exception:
            return np.nan
    df["parity_gap"] = df.apply(parity, axis=1)

    clean = df[
        (df["days_expiry_after_earnings"] <= MAX_EXPIRY_GAP) &
        (df["price_source"] == "mid") &
        (df["parity_gap"].fillna(1) <= 0.01)
    ].copy()
    print(f"Clean sample after quality filters: {len(clean)} "
          f"(dropped {len(df) - len(clean)})")
    return clean


def attach_history(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each ticker's most recent trailing MEAN historical EM."""
    ev = pd.read_csv(EVENT_EM, parse_dates=["reaction_date"])
    if "hist_em_mean_8q" not in ev.columns:
        raise SystemExit("event_em.csv missing hist_em_mean_8q — re-run realized_em.py")
    ev = ev.dropna(subset=["hist_em_mean_8q"]).sort_values("reaction_date")
    hist = ev.groupby("ticker")["hist_em_mean_8q"].last()

    df = df.copy()
    df["hist_em"] = df["ticker"].map(hist)
    before = len(df)
    df = df.dropna(subset=["hist_em", "implied_em"])
    print(f"Events with matched history: {len(df)} (dropped {before - len(df)} "
          f"with no historical EM)")
    return df


def attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    con = pd.read_csv(CONSTITUENTS)
    con["Symbol"] = con["Symbol"].astype(str).str.replace(".", "-", regex=False)
    sec = con.set_index("Symbol")["GICS Sector"]
    df = df.copy()
    df["sector"] = df["ticker"].map(sec)
    return df


# ---------------------------------------------------------------------------
def calibrate(df: pd.DataFrame) -> dict:
    x = df["hist_em"].values * 100      # historical EM (%)
    y = df["implied_em"].values * 100   # implied EM (%)

    spearman = stats.spearmanr(x, y).correlation
    pearson = stats.pearsonr(x, y)[0]
    slope, intercept, r, p, se = stats.linregress(x, y)
    ratio = df["implied_em"] / df["hist_em"]

    return {
        "n": len(df), "spearman": spearman, "pearson": pearson, "r2": r ** 2,
        "slope": slope, "intercept": intercept,
        "median_ratio": ratio.median(), "mean_ratio": ratio.mean(),
    }


def make_scatter(df: pd.DataFrame, c: dict):
    x = df["hist_em"] * 100
    y = df["implied_em"] * 100
    lim = max(x.max(), y.max()) * 1.08

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=22, color=INK, alpha=0.55, zorder=3)
    ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1.3,
            label="implied = historical", zorder=1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, c["slope"] * xs + c["intercept"], color=ACCENT, lw=2,
            label=f"fit: slope={c['slope']:.2f}", zorder=2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Historical expected move (trailing mean, %)")
    ax.set_ylabel("Implied expected move (%)")
    ax.set_title(f"The options market anchors implied EM to history\n"
                 f"Spearman = {c['spearman']:.2f},  R\u00b2 = {c['r2']:.2f},  n = {c['n']}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_calibration.png")
    plt.close(fig)


def make_sector_figure(df: pd.DataFrame):
    g = (df.dropna(subset=["sector"])
           .groupby("sector")
           .apply(lambda s: stats.spearmanr(s["hist_em"], s["implied_em"]).correlation
                  if len(s) >= 5 else np.nan, include_groups=False)
           .dropna().sort_values())
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(g.index, g.values, color=INK, alpha=0.85)
    ax.set_xlabel("Spearman(implied, historical) within sector")
    ax.set_title("Calibration tightness by GICS sector (sectors with n\u22655)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_calibration_by_sector.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    ok = load_snapshots()
    pre = primary_events(ok)
    clean = quality_filter(pre)
    df = attach_history(clean)
    df = attach_sector(df)

    df["divergence"] = df["implied_em"] / df["hist_em"]
    df.to_csv(OUT_TABLE, index=False)

    c = calibrate(df)
    make_scatter(df, c)
    make_sector_figure(df)

    lines = []
    def P(s=""):
        lines.append(s); print(s)

    P("=" * 60)
    P("SECTION 6.1 — CALIBRATION: IMPLIED EM vs HISTORICAL EM")
    P("=" * 60)
    P(f"Clean events analyzed:     {c['n']}")
    P(f"Spearman (rank):           {c['spearman']:.3f}")
    P(f"Pearson:                   {c['pearson']:.3f}")
    P(f"R-squared:                 {c['r2']:.3f}")
    P(f"Regression slope:          {c['slope']:.3f}")
    P(f"Regression intercept:      {c['intercept']:.3f} (pp)")
    P(f"Median implied/historical: {c['median_ratio']:.3f}")
    P(f"Mean implied/historical:   {c['mean_ratio']:.3f}")
    P("")
    P("Interpretation:")
    P(f"  slope {c['slope']:.2f}: each 1pp of historical EM maps to "
      f"{c['slope']:.2f}pp of implied EM")
    P(f"  median ratio {c['median_ratio']:.2f}: the typical straddle prices "
      f"{(c['median_ratio']-1)*100:+.0f}% vs history")
    P("")
    P("LARGEST DIVERGENCES (implied >> historical) — Section 6.2 candidates:")
    top = df.nlargest(8, "divergence")[["ticker", "implied_em", "hist_em", "divergence"]]
    for _, r in top.iterrows():
        P(f"  {r.ticker:6s} implied={r.implied_em*100:5.2f}%  "
          f"hist={r.hist_em*100:5.2f}%  ratio={r.divergence:.2f}")
    P("")
    P("LARGEST NEGATIVE DIVERGENCES (implied << historical):")
    bot = df.nsmallest(8, "divergence")[["ticker", "implied_em", "hist_em", "divergence"]]
    for _, r in bot.iterrows():
        P(f"  {r.ticker:6s} implied={r.implied_em*100:5.2f}%  "
          f"hist={r.hist_em*100:5.2f}%  ratio={r.divergence:.2f}")
    P("=" * 60)
    P("Figures -> report/figures/fig6_calibration.png, fig7_calibration_by_sector.png")
    P(f"Per-event table -> {OUT_TABLE}")

    (TABDIR / "part2_calibration.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()