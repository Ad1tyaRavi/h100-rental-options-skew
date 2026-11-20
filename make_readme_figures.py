#!/usr/bin/env python
"""Two supporting figures for the README (about 15 s, CPU, fixed seed).

    python make_readme_figures.py

outputs/ornn_h100_history.png         the Ornn H100 SXM index with the 0.10 markdown for scale
outputs/settlement_distributions.png  why the two settlements skew differently (persistent 0.10 case)
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from h100_skew_demo import BASE, SEED, ORNN_RAW, OUT, build_block, settlements, smile

RED, BLUE, GREY, INK, MUTED = "#b2182b", "#2166ac", "#9a9a9a", "#222222", "#666666"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bbbbbb",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})


def ornn_figure():
    recs = sorted(json.load(open(ORNN_RAW, encoding="utf-8"))["data"], key=lambda r: r["timestamp"])
    d = np.array([np.datetime64(r["timestamp"][:10]) for r in recs])
    v = np.array([float(r["index_value"]) for r in recs])
    lv = np.log(v)
    i1 = int(np.argmin(np.diff(lv))) + 1                      # largest one-day fall lands on day i1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d, v, color=INK, lw=1.6)
    ax.plot(d[[i1 - 1, i1]], v[[i1 - 1, i1]], color=RED, lw=2.6)
    ax.annotate(f"largest one-day fall\n{v[i1 - 1]:.2f} → {v[i1]:.2f} ({100 * (v[i1] / v[i1 - 1] - 1):.1f}%), "
                f"back to {v[i1 + 4]:.2f} four days later",
                xy=(d[i1], v[i1]), xytext=(d[i1] + np.timedelta64(6, "D"), 2.40),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    last, cut = v[-1], v[-1] * math.exp(-0.10)
    x_arrow = d[-1] + np.timedelta64(3, "D")
    ax.annotate("", xy=(x_arrow, cut), xytext=(x_arrow, last),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
    ax.text(x_arrow + np.timedelta64(2, "D"), (last + cut) / 2,
            f"the 0.10 threshold markdown\nfrom the last print: {last:.2f} → {cut:.2f}\n(it has to stay there)",
            fontsize=8.5, color=INK, va="center")
    ax.set_xlim(d[0] - np.timedelta64(2, "D"), d[-1] + np.timedelta64(30, "D"))
    ax.set_ylim(2.3, 3.65)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("$ per GPU-hour")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_title(f"Ornn H100 SXM daily index, {d[0]} to {d[-1]} ({len(v)} prints, free-tier window)",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ornn_h100_history.png"), dpi=160)
    plt.close(fig)


def distribution_figure():
    blocks = [build_block(SEED + i, 25_000, BASE) for i in range(8)]
    cases = {"no launch": settlements(blocks, 0.0, math.inf, 0.5, 1, 30, BASE["mag_cv"]),
             "launch": settlements(blocks, 0.10, math.inf, 0.5, 1, 30, BASE["mag_cv"])}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    spec = ((axes[0], 0, RED, "Final daily print", 0.80,
             (0.52, "upward spikes land in full:\nlong right tail, call skew")),
            (axes[1], 1, BLUE, "Monthly average", 0.40,
             (-0.395, "spikes are averaged away;\nan unknown launch date leaves\na left shoulder: put skew")))
    for ax, j, col, name, lim, (xt, note) in spec:
        for label, setts in cases.items():
            x = np.concatenate([s[j] for s in setts])
            rr = 100 * smile(x)["rr25"]
            z = np.log(x / x.mean())
            h, e = np.histogram(z, bins=np.linspace(-lim, lim, 81), density=True)
            mid = 0.5 * (e[1:] + e[:-1])
            h = np.where(h > 0, h, np.nan)
            if label == "launch":
                ax.plot(mid, h, color=col, lw=2.0, label=f"50% chance of a 0.10 markdown that persists   RR {rr:+.1f}")
            else:
                ax.plot(mid, h, color=GREY, lw=1.5, ls="--", label=f"no launch   RR {rr:+.1f}")
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 30)
        ax.set_xlim(-lim, lim)
        ax.set_yticks([])
        ax.minorticks_off()
        ax.set_xlabel("log(settlement / its own forward)")
        ax.set_ylabel("density, log scale", color=MUTED)
        ax.set_title(name, fontsize=11, loc="left", color=INK)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        ax.text(xt, 1.2 if j == 0 else 2.0, note, fontsize=8.5, color=INK, ha="center" if j == 0 else "left")
    fig.suptitle("Same 200,000 simulated months, two settlements (RR = 25Δ risk reversal, call IV − put IV, vol points)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "settlement_distributions.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ornn_figure()
    distribution_figure()
    print("saved outputs/ornn_h100_history.png and outputs/settlement_distributions.png")
