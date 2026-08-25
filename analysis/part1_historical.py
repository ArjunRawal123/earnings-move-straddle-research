"""
part1_historical.py — Section 5 (The Historical Anchor): final statistics + figures.

Turns the terminal one-liners into reproducible, saved artifacts. Loads the
scored event table once, recomputes every Section 5 number, writes four figures
to report/figures/, and prints a paste-ready summary table.

Run from the EM-Research project root:
    python3 analysis/part1_historical.py

Inputs : data/event_em.csv        (from realized_em.py, timing-corrected)
         data/constituents.csv     (GICS sectors, for the breakdown)
Outputs: report/figures/fig1_unbiasedness.png
         report/figures/fig2_durability.png
         report/figures/fig3_within_stock_null.png
         report/figures/fig4_sector_breakdown.png
         report/tables/part1_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")                 # no display needed; write straight to file
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVENT_EM = ROOT / "data" / "event_em.csv"
CONSTITUENTS = ROOT / "data" / "constituents.csv"
FIGDIR = ROOT / "report" / "figures"
TABDIR = ROOT / "report" / "tables"

PRIMARY_WINDOW = 8                    # headline trailing window (quarters)
MIN_TICKER_EVENTS = 20               # for per-ticker correlations

# Consistent plain styling for a written report (no seaborn dependency)
plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
INK = "#1f2d3d"
ACCENT = "#c0392b"


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    d = pd.read_csv(EVENT_EM, parse_dates=["reaction_date"])
    print(f"Loaded {len(d):,} scored events | {d.ticker.nunique()} tickers")
    return d


# ---------------------------------------------------------------------------
# 5.1 — Unbiasedness
# ---------------------------------------------------------------------------
def unbiasedness(d: pd.DataFrame) -> dict:
    col = f"em_ratio_{PRIMARY_WINDOW}q"
    r = d[col].dropna()
    stats = {
        "n": len(r),
        "median_ratio": r.median(),
        "mean_ratio": r.mean(),
        "share_within_50pct": ((r > 0.5) & (r < 1.5)).mean(),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    clipped = r.clip(upper=5)
    ax.hist(clipped, bins=80, color=INK, alpha=0.85)
    ax.axvline(1.0, color=ACCENT, lw=2, label="ratio = 1 (unbiased)")
    ax.axvline(r.median(), color="black", ls="--", lw=1.2,
               label=f"median = {r.median():.3f}")
    ax.set_xlabel(f"Realized move ÷ trailing {PRIMARY_WINDOW}-quarter median EM")
    ax.set_ylabel("Number of events")
    ax.set_title("Historical EM is an unbiased forecast of the realized move")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_unbiasedness.png")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 5.2 — Durability (split-half)
# ---------------------------------------------------------------------------
def durability(d: pd.DataFrame) -> dict:
    d = d.sort_values(["ticker", "reaction_date"]).copy()
    d["frac"] = d.groupby("ticker").cumcount() / d.groupby("ticker")["abs_move"].transform("size")
    a = d[d.frac < 0.5].groupby("ticker")["abs_move"].median()
    b = d[d.frac >= 0.5].groupby("ticker")["abs_move"].median()
    m = pd.concat([a, b], axis=1, keys=["first_half", "second_half"]).dropna()
    rho = m["first_half"].corr(m["second_half"], method="spearman")

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(m.first_half * 100, m.second_half * 100, s=14, color=INK, alpha=0.55)
    lim = [0, max(m.max()) * 100 * 1.05]
    ax.plot(lim, lim, color=ACCENT, lw=1.5, label="45° (perfect persistence)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Median |move|, first half of history (%)")
    ax.set_ylabel("Median |move|, second half (%)")
    ax.set_title(f"A stock's earnings-move level is durable\nSpearman = {rho:.3f}  (n = {len(m)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_durability.png")
    plt.close(fig)
    return {"n": len(m), "spearman": rho}


# ---------------------------------------------------------------------------
# 5.3 — No within-stock timing power
# ---------------------------------------------------------------------------
def within_stock(d: pd.DataFrame) -> dict:
    col = f"hist_em_{PRIMARY_WINDOW}q"
    d = d.sort_values(["ticker", "reaction_date"]).copy()

    # pooled and demeaned, single-next-event
    s = d.dropna(subset=[col, "abs_move"])
    pooled = s[col].corr(s["abs_move"], method="spearman")
    sd = s.copy()
    for c in [col, "abs_move"]:
        sd[c + "_dm"] = sd[c] - sd.groupby("ticker")[c].transform("median")
    within = sd[col + "_dm"].corr(sd["abs_move_dm"], method="spearman")

    # per-ticker correlations + excess-variance test
    rows = []
    for t, g in d.dropna(subset=[col]).groupby("ticker"):
        if len(g) >= MIN_TICKER_EVENTS and g[col].nunique() > 1:
            rows.append((t, len(g), g[col].corr(g["abs_move"], method="spearman")))
    pt = pd.DataFrame(rows, columns=["ticker", "n", "rho"]).dropna()
    obs_sd = pt["rho"].std()
    exp_sd = np.mean(1 / np.sqrt(pt["n"] - 1))
    excess = obs_sd / exp_sd

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(pt["rho"], bins=30, color=INK, alpha=0.85, density=True,
            label=f"observed (SD={obs_sd:.3f})")
    xs = np.linspace(-1, 1, 400)
    ax.plot(xs, (1 / (exp_sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs / exp_sd) ** 2),
            color=ACCENT, lw=2, label=f"pure noise (SD={exp_sd:.3f})")
    ax.axvline(pt["rho"].mean(), color="black", ls="--", lw=1.2,
               label=f"mean = {pt['rho'].mean():.3f}")
    ax.set_xlabel(f"Per-ticker Spearman(trailing {PRIMARY_WINDOW}q EM, realized move)")
    ax.set_ylabel("Density")
    ax.set_title("History does not predict a stock's next move vs. its own norm")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_within_stock_null.png")
    plt.close(fig)

    return {
        "pooled": pooled, "within": within,
        "per_ticker_mean": pt["rho"].mean(), "n_tickers": len(pt),
        "obs_sd": obs_sd, "exp_sd": exp_sd, "excess_ratio": excess,
        "share_positive": (pt["rho"] > 0).mean(),
    }


# ---------------------------------------------------------------------------
# Sector breakdown (supporting figure)
# ---------------------------------------------------------------------------
def sector_breakdown(d: pd.DataFrame) -> pd.DataFrame:
    con = pd.read_csv(CONSTITUENTS)
    con["Symbol"] = con["Symbol"].astype(str).str.replace(".", "-", regex=False)
    sec = con.set_index("Symbol")["GICS Sector"]
    d = d.copy()
    d["sector"] = d["ticker"].map(sec)
    g = (d.dropna(subset=["sector"])
           .groupby("sector")["abs_move"].median().mul(100).sort_values())

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(g.index, g.values, color=INK, alpha=0.85)
    ax.set_xlabel("Median absolute earnings move (%)")
    ax.set_title("Earnings-move magnitude by GICS sector")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_sector_breakdown.png")
    plt.close(fig)
    return g


# ---------------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    d = load()
    desc = {
        "median_move": d["abs_move"].median(),
        "mean_move": d["abs_move"].mean(),
        "p95_move": d["abs_move"].quantile(0.95),
    }
    u = unbiasedness(d)
    dur = durability(d)
    ws = within_stock(d)
    sec = sector_breakdown(d)

    lines = []
    def P(s=""):
        lines.append(s)
        print(s)

    P("=" * 64)
    P("SECTION 5 — THE HISTORICAL ANCHOR — SUMMARY")
    P("=" * 64)
    P(f"Events scored:            {len(d):,}   ({d.ticker.nunique()} tickers)")
    P(f"Median |move|:            {desc['median_move']*100:.2f}%")
    P(f"Mean |move|:              {desc['mean_move']*100:.2f}%")
    P(f"95th percentile |move|:   {desc['p95_move']*100:.2f}%")
    P("")
    P("5.1  UNBIASEDNESS")
    P(f"     median(realized / hist_{PRIMARY_WINDOW}q)  = {u['median_ratio']:.3f}   (n={u['n']:,})")
    P(f"     mean(realized / hist_{PRIMARY_WINDOW}q)    = {u['mean_ratio']:.3f}")
    P(f"     share within +/-50% of forecast    = {u['share_within_50pct']*100:.1f}%")
    P("")
    P("5.2  DURABILITY (split-half, ticker medians)")
    P(f"     Spearman(first half, second half)  = {dur['spearman']:.3f}   (n={dur['n']})")
    P("")
    P("5.3  WITHIN-STOCK TIMING")
    P(f"     pooled single-event Spearman       = {ws['pooled']:.3f}")
    P(f"     within-stock (demeaned)            = {ws['within']:.3f}")
    P(f"     per-ticker mean Spearman           = {ws['per_ticker_mean']:.3f}   (n={ws['n_tickers']})")
    P(f"     share of tickers positive          = {ws['share_positive']*100:.1f}%")
    P(f"     excess-variance ratio              = {ws['excess_ratio']:.2f}")
    P(f"       (observed SD {ws['obs_sd']:.3f} vs noise SD {ws['exp_sd']:.3f})")
    P("")
    P("SECTOR MEDIAN |move| (%)")
    for k, v in sec.items():
        P(f"     {k:24s} {v:5.2f}")
    P("=" * 64)
    P("Figures written to report/figures/  (fig1..fig4)")

    (TABDIR / "part1_summary.txt").write_text("\n".join(lines))
    print(f"\nSummary table -> {TABDIR / 'part1_summary.txt'}")


if __name__ == "__main__":
    main()