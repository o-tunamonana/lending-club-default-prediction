"""Fairness check: does the champion model treat groups of applicants differently?

Trains the tuned gradient boosting model (same settings as the registered champion),
sets the approval cutoff on the training period (safest 80% approved), then scores the
unseen 2016 cohort and compares groups by home ownership and by state:

- approval rate and the ratio to the most-approved group (the "80% rule")
- actual default rate vs. mean predicted probability (calibration gap)
- default rate among approved loans
- false rejection rate: good borrowers (no default) who are rejected
- missed default rate: borrowers who default but are approved

Run from the repo root (needs src/lcdp/train.py):  PYTHONPATH=src python scripts/fairness_analysis.py
Writes docs/fairness.md and docs/images/fairness-*.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lcdp.schema import FEATURES, TARGET  # noqa: E402
from lcdp.train import approval_cutoff, make_candidates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MIN_GROUP = 500          # smaller groups are too noisy to compare
TOP_STATES = 12          # largest states by number of 2016 loans

BLUE, GRAY, TEXT, MUTED = "#257de0", "#b8b8b8", "#191919", "#666666"


def group_metrics(df: pd.DataFrame, col: str, cutoff: float) -> pd.DataFrame:
    d = df.assign(approved=df["proba"] <= cutoff)

    def one(g):
        good, bad = g[g[TARGET] == 0], g[g[TARGET] == 1]
        return pd.Series({
            "loans": len(g),
            "actual_default": g[TARGET].mean(),
            "mean_predicted": g["proba"].mean(),
            "approval_rate": g["approved"].mean(),
            "default_if_approved": g.loc[g["approved"], TARGET].mean(),
            "false_rejection": (~good["approved"]).mean(),
            "missed_default": bad["approved"].mean(),
        })

    out = d.groupby(col).apply(one, include_groups=False)
    out = out[out["loans"] >= MIN_GROUP].sort_values("loans", ascending=False)
    out["approval_ratio"] = out["approval_rate"] / out["approval_rate"].max()
    out["calibration_gap"] = out["mean_predicted"] - out["actual_default"]
    overall = one(d)
    overall["approval_ratio"] = float("nan")
    overall["calibration_gap"] = overall["mean_predicted"] - overall["actual_default"]
    out.loc["All loans"] = overall
    out["loans"] = out["loans"].astype(int)
    return out


def to_markdown(t: pd.DataFrame, label: str) -> str:
    pct = ["actual_default", "mean_predicted", "calibration_gap", "approval_rate",
           "default_if_approved", "false_rejection", "missed_default"]
    head = (f"| {label} | Loans | Actual default | Mean predicted | Gap | Approval | "
            "Approval ratio | Default if approved | False rejection | Missed default |")
    lines = [head, "|" + "---|" * 10]
    for name, r in t.iterrows():
        cells = [f"{r[c]:+.1%}" if c == "calibration_gap" else f"{r[c]:.1%}" for c in pct]
        ratio = "" if pd.isna(r["approval_ratio"]) else f"{r['approval_ratio']:.2f}"
        lines.append(f"| {name} | {int(r['loans']):,} | {cells[0]} | {cells[1]} | {cells[2]} | "
                     f"{cells[3]} | {ratio} | {cells[4]} | {cells[5]} | {cells[6]} |")
    return "\n".join(lines)


def plot(t: pd.DataFrame, title: str, path: Path) -> None:
    """Small multiples (one shared scale per panel, no dual axes)."""
    t = t.drop(index="All loans")
    panels = [("approval_rate", "Approval rate"),
              ("false_rejection", "Good borrowers rejected"),
              ("missed_default", "Defaulters approved"),
              ("calibration_gap", "Predicted − actual default (pt)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.42 * len(t) + 1.6), sharey=True)
    names = [str(i) for i in t.index][::-1]
    for ax, (col, label) in zip(axes, panels, strict=True):
        vals = t[col].values[::-1] * 100
        ax.barh(names, vals, color=BLUE, height=0.6)
        ax.axvline(0, color=GRAY, lw=1)
        for y, v in enumerate(vals):
            ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:.0f}%" if col != "calibration_gap"
                    else f"{v:+.1f} pt", va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, color=TEXT)
        ax.set_title(label, fontsize=11, color=TEXT, loc="left")
        ax.tick_params(colors=MUTED, labelsize=10, length=0)
        ax.xaxis.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
        lo, hi = min(vals.min(), 0), max(vals.max(), 0)
        pad = (hi - lo) * 0.25 + 1
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    model, _ = make_candidates()["gradient_boosting_tuned"]
    model.fit(train[FEATURES], train[TARGET].astype(int))
    cutoff = approval_cutoff(model.predict_proba(train[FEATURES])[:, 1])
    test = test.assign(proba=model.predict_proba(test[FEATURES])[:, 1])

    home = group_metrics(test, "home_ownership_n", cutoff)
    top = test["addr_state"].value_counts().head(TOP_STATES).index
    state = group_metrics(test[test["addr_state"].isin(top)], "addr_state", cutoff)
    state.loc["All loans"] = home.loc["All loans"]   # overall row = whole 2016 cohort

    (DOCS / "images").mkdir(parents=True, exist_ok=True)
    plot(home, "By home ownership (2016 test cohort)", DOCS / "images/fairness-home.png")
    plot(state, f"By state, {TOP_STATES} largest (2016 test cohort)",
         DOCS / "images/fairness-state.png")

    (DOCS / "fairness_tables.md").write_text(
        f"Cutoff: approve if predicted default probability ≤ {cutoff:.3f} "
        f"(safest 80% of the training period). Test cohort: {len(test):,} loans issued in 2016.\n\n"
        f"### By home ownership\n\n{to_markdown(home, 'Home ownership')}\n\n"
        f"### By state ({TOP_STATES} largest)\n\n{to_markdown(state, 'State')}\n"
    )
    print(home.round(3).to_string(), "\n\n", state.round(3).to_string())


if __name__ == "__main__":
    main()
