#!/usr/bin/env python
"""Conditional repricing of the two hypothetical H100 options through the month.

    python h100_conditional_demo.py        (about 30 s, CPU, fixed seed)

Builds on h100_skew_demo.py: baseline spot model, persistent 0.10 log markdown,
launch probability 0.5, launch day uniform on 1..30, day-30 settlement.

At each valuation day t the launch has NOT happened yet. Past prints are an
explicit illustrative history and stay fixed in every continuation; the spot
factor state at day t is carried into the remaining simulation. The launch is
updated by Bayes' rule inside the existing model:

    P(launch in days t+1..30 | none by t) = p*(30-t)/30 / (1 - p*t/30)

with the launch day uniform on the remaining days. A control holds that
probability at its day-zero value p with the same remaining-date distribution.
One extra branch: the launch lands on day 10 and its markdown is observed.

Outputs: outputs/conditional_repricing_figure.png, outputs/conditional_repricing.json
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h100_skew_demo import BASE, SEED, N_DAYS, Z_CRIT, OUT, scenario_stats, clean

M_MEAN = 0.10            # mean log markdown, persistent (the original threshold case)
VAL_DAYS = [0, 5, 10, 15, 20, 25]   # 0, 10, 20 are the requested dates; the others fill in the chart
N_BLOCKS, N_PER = 16, 25_000
HIST_SEED = SEED + 777
OBSERVED_MARKDOWN = 0.10  # day-10 branch: the launch lands on day 10 at exactly the mean size


def make_history(par, n_hist, seed):
    """One explicit illustrative no-launch history with its factor states.
    No spike occurs in the history (spike state stays 0); the fast-noise and
    slow-level states are seeded draws from the baseline model."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / par["ou_hl"])
    x = r = 0.0
    prices, states = [], [(0.0, 0.0, 0.0)]
    for _ in range(n_hist):
        x = phi * x + par["ou_sd"] * rng.standard_normal()
        r = r + par["rw_sd"] * rng.standard_normal()
        prices.append(par["S0"] * math.exp(x + r))
        states.append((x, 0.0, r))
    return np.array(prices), states          # states[t] = (x, y, r) after the day-t print


def continue_block(seed, n, state, t0, par):
    """Launch-free continuation for days t0+1..30 from the day-t0 factor state."""
    nrem = N_DAYS - t0
    rng = np.random.default_rng(seed)
    zx = rng.standard_normal((n, nrem))
    zr = rng.standard_normal((n, nrem))
    us = rng.random((n, nrem))
    es = rng.exponential(size=(n, nrem))
    blk = dict(u_occ=rng.random(n), u_day=rng.random(n), z_mag=rng.standard_normal(n))
    phi_x, phi_y = 0.5 ** (1.0 / par["ou_hl"]), 0.5 ** (1.0 / par["spk_hl"])
    jump = (us < par["spk_lam"] / 365.0) * es * par["spk_mean"]
    x = np.full(n, state[0])
    y = np.full(n, state[1])
    r = state[2] + np.cumsum(par["rw_sd"] * zr, axis=1)
    logb = np.empty((n, nrem))
    for t in range(nrem):
        x = phi_x * x + par["ou_sd"] * zx[:, t]
        y = phi_y * y + jump[:, t]
        logb[:, t] = x + y + r[:, t]
    blk["base"] = par["S0"] * np.exp(logb)
    return blk


def settle(blk, past_prints, p_rem, par, existing_markdown=0.0):
    """Final print and full-month average; past prints are fixed numbers."""
    n, nrem = blk["base"].shape
    logm = np.full((n, nrem), -existing_markdown)
    if p_rem > 0:
        occ = blk["u_occ"] < p_rem
        tau = np.floor(blk["u_day"] * nrem).astype(int)          # uniform on the remaining days
        cv = par["mag_cv"]
        mag = M_MEAN * np.exp(cv * blk["z_mag"] - 0.5 * cv * cv)
        logm = logm - mag[:, None] * (np.arange(nrem)[None, :] >= tau[:, None]) * occ[:, None]
    s = blk["base"] * np.exp(logm)
    return s[:, -1].copy(), (past_prints.sum() + s.sum(axis=1)) / N_DAYS


def price(blocks, past_prints, p_rem, t0, existing_markdown=0.0):
    st = scenario_stats([settle(b, past_prints, p_rem, BASE, existing_markdown) for b in blocks])
    # scenario_stats annualises with the full 30 days; the 25-delta strikes depend only on
    # total vol, so rescaling converts every vol to the remaining time to expiry.
    scale = math.sqrt(N_DAYS / (N_DAYS - t0))
    out = {}
    for name in ("fin", "avg"):
        for key in ("rr25", "rrk", "atm"):
            out[f"{key}_{name}"] = st[f"{key}_{name}"] * scale
            out[f"{key}_{name}_se"] = st[f"{key}_{name}_se"] * scale
        out[f"fwd_{name}"] = st[f"fwd_{name}"]
        out[f"flags_{name}"] = st[f"flags_{name}"]
        out[f"rr_over_atm_{name}"] = st[f"rr25_{name}"] / st[f"atm_{name}"]
    out["flip"] = st["flip"]
    out["p_remaining"] = p_rem
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    p = BASE["p"]
    hist, states = make_history(BASE, max(VAL_DAYS), HIST_SEED)
    res = dict(seed=SEED, history_seed=HIST_SEED, markdown_mean=M_MEAN, paths=N_BLOCKS * N_PER,
               history_prints=[round(float(v), 4) for v in hist],
               history_states=[dict(day=i, x=s[0], y=s[1], r=s[2]) for i, s in enumerate(states)],
               updated={}, held={})
    print("illustrative history ($/GPU-hour), days 1..%d:" % len(hist))
    print("  " + " ".join(f"{v:.2f}" for v in hist))

    for t0 in VAL_DAYS:
        blocks = [continue_block(SEED + 9000 + 100 * t0 + i, N_PER, states[t0], t0, BASE) for i in range(N_BLOCKS)]
        p_upd = p * (N_DAYS - t0) / N_DAYS / (1.0 - p * t0 / N_DAYS)
        res["updated"][t0] = price(blocks, hist[:t0], p_upd, t0)
        res["held"][t0] = price(blocks, hist[:t0], p, t0)
        if t0 == 10:
            # launch lands on day 10 with an observed markdown; same prints for days 1-9,
            # same factor state, no further launch this month
            past = hist[:10].copy()
            past[9] *= math.exp(-OBSERVED_MARKDOWN)
            res["launched_day10"] = price(blocks, past, 0.0, t0, existing_markdown=OBSERVED_MARKDOWN)
            res["launched_day10"]["day10_print"] = float(past[9])

    def row(tag, t0, r):
        return (f"{tag:<26} day {t0:>2}  P(launch left)={r['p_remaining']:.3f}  "
                f"fwd {r['fwd_fin']:.3f}/{r['fwd_avg']:.3f}  ATM {100 * r['atm_fin']:5.1f}/{100 * r['atm_avg']:5.1f}  "
                f"RR25 final {100 * r['rr25_fin']:+6.2f}+/-{100 * r['rr25_fin_se']:.2f}  "
                f"avg {100 * r['rr25_avg']:+6.2f}+/-{100 * r['rr25_avg_se']:.2f}  "
                f"RRk {100 * r['rrk_fin']:+6.2f}/{100 * r['rrk_avg']:+6.2f}  "
                f"RR/ATM {r['rr_over_atm_fin']:+.3f}/{r['rr_over_atm_avg']:+.3f}  flip={r['flip']} "
                f"flags={r['flags_fin'] + r['flags_avg']}")
    for t0 in VAL_DAYS:
        print(row("no launch yet, updated", t0, res["updated"][t0]))
        print(row("no launch yet, P held", t0, res["held"][t0]))
    print(row("launch observed day 10", 10, res["launched_day10"]))

    # ---- chart -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    days = np.array(VAL_DAYS)
    series = (("fin", "#b2182b", "final daily print"), ("avg", "#2166ac", "monthly average"))
    for ax, mode in ((axes[0], "vol"), (axes[1], "ratio")):
        for name, col, lab in series:
            for src, ls, mk, fill, tag in (("updated", "-", "o", col, "probability updated"),
                                           ("held", "--", "s", "white", "probability held at 0.50")):
                if mode == "vol":
                    y = np.array([100 * res[src][t][f"rr25_{name}"] for t in VAL_DAYS])
                    e = np.array([100 * Z_CRIT * res[src][t][f"rr25_{name}_se"] for t in VAL_DAYS])
                    ax.errorbar(days, y, yerr=e, color=col, ls=ls, marker=mk, mfc=fill, ms=6, capsize=3,
                                lw=1.7, label=f"{lab}, {tag}")
                else:
                    y = np.array([res[src][t][f"rr_over_atm_{name}"] for t in VAL_DAYS])
                    ax.plot(days, y, color=col, ls=ls, marker=mk, mfc=fill, ms=6, lw=1.7, label=f"{lab}, {tag}")
            b = res["launched_day10"]
            yb = 100 * b[f"rr25_{name}"] if mode == "vol" else b[f"rr_over_atm_{name}"]
            ax.plot([10], [yb], marker="*", ms=15, color=col, mec="black", ls="none",
                    label=f"{lab}, launch observed on day 10")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(days, [f"day {t}\nP={res['updated'][t]['p_remaining']:.2f}" for t in VAL_DAYS])
        ax.set_xlabel("valuation day (launch not yet seen); P = updated probability of a launch in the days left")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("25Δ risk reversal, vol points (remaining time to expiry)")
    axes[0].set_title(f"Risk reversal (call IV − put IV), ±{Z_CRIT:g} s.e.", fontsize=11)
    axes[1].set_ylabel("25Δ risk reversal ÷ ATM vol")
    axes[1].set_title("Same, scaled by each option's ATM vol", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Repricing through the month while the launch has not happened: persistent 0.10 log markdown, "
                 "day-30 settlement, one fixed illustrative price history\n"
                 "(hypothetical options, assumed pricing measure; average-option vols are quoted on the whole-month "
                 "average with past prints fixed)", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(os.path.join(OUT, "conditional_repricing_figure.png"), dpi=160)
    with open(os.path.join(OUT, "conditional_repricing.json"), "w") as f:
        json.dump(clean(res), f, indent=1)
    print("saved figure and json in", OUT)


if __name__ == "__main__":
    main()
