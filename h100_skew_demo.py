#!/usr/bin/env python
"""H100 rental-price option skew: final-print vs monthly-average settlement.

One command, CPU only, fixed seed:

    python h100_skew_demo.py            # full run (~2-4 min)
    python h100_skew_demo.py --quick    # smoke test (~20 s, noisier)

Dependencies: numpy, scipy, matplotlib (standard scientific stack).

Outputs (written to ./outputs):
    skew_main_figure.png      main figure (risk reversals over markdown size x persistence)
    grid_results.csv          every cell of the main sweep
    sensitivity_results.csv   threshold under timing / probability / spike changes
    summary.json              headline numbers, controls, verification, Ornn statistics

Everything is specified directly under an ASSUMED pricing measure Q with zero
discounting. Both options are hypothetical. See RESEARCH_NOTE.md.
"""
import argparse
import csv
import json
import math
import os
import time
import urllib.request

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ORNN_RAW = os.path.join(HERE, "data", "ornn_raw", "h100_sxm_index_history.json")
ORNN_URL = "https://api.ornnai.com/api/gpu/H100%20SXM/index-history"

SEED = 20261116
N_DAYS = 30                 # settlement month approximated by 30 daily prints
T_YEARS = N_DAYS / 365.0    # option expiry = last print of the month
Z_CRIT = 2.0                # a sign is "credible" if |RR| > Z_CRIT * MC standard error

# ----------------------------------------------------------------------------
# Baseline parameters (all in natural-log units per daily print, under Q)
# ----------------------------------------------------------------------------
BASE = dict(
    S0=3.00,          # $/GPU-hour, close to the Ornn H100 SXM level in the sample
    ou_sd=0.042,      # daily innovation s.d. of the fast mean-reverting deviation (matches Ornn median |1d| move)
    ou_hl=1.0,        # its half-life in days
    rw_sd=0.012,      # daily s.d. of the slow random-walk level
    spk_lam=12.0,     # upward scarcity spikes per year (Lee et al. Table 4 baseline)
    spk_mean=0.15,    # mean log spike size (exponential => Pareto tail index ~6.7)
    spk_hl=2.0,       # spike half-life in days (Lee et al.: 1/kappa_Y ~ 2.8 days)
    p=0.5,            # probability that a launch markdown lands in the month
    day_lo=1,         # launch day ~ Uniform{day_lo..day_hi}
    day_hi=30,
    mag_cv=0.30,      # lognormal dispersion of markdown size around its mean
)
# Sweep ranges for the main figure
SIZES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]   # mean log markdown
HALFLIVES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0, math.inf]       # days; inf = persists
SENS_HALFLIVES = [2.0, 5.0, 10.0, 30.0, math.inf]


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def build_block(seed, n, par):
    """Launch-free price paths plus the launch random numbers for one block.

    The draw order is fixed, so the same seed gives common random numbers across
    every scenario (markdown size, persistence, timing, probability, spike decay).
    """
    rng = np.random.default_rng(seed)
    zx = rng.standard_normal((n, N_DAYS))
    zr = rng.standard_normal((n, N_DAYS))
    us = rng.random((n, N_DAYS))
    es = rng.exponential(size=(n, N_DAYS))
    u_occ = rng.random(n)
    u_day = rng.random(n)
    z_mag = rng.standard_normal(n)

    phi_x = 0.5 ** (1.0 / par["ou_hl"])
    phi_y = 0.5 ** (1.0 / par["spk_hl"])
    jump = (us < par["spk_lam"] / 365.0) * es * par["spk_mean"]
    x = np.zeros(n)
    y = np.zeros(n)
    logb = np.empty((n, N_DAYS))
    r = np.cumsum(par["rw_sd"] * zr, axis=1)
    for t in range(N_DAYS):
        x = phi_x * x + par["ou_sd"] * zx[:, t]
        y = phi_y * y + jump[:, t]
        logb[:, t] = x + y + r[:, t]
    base = par["S0"] * np.exp(logb)
    return dict(base=base, u_occ=u_occ, u_day=u_day, z_mag=z_mag)


def markdown_log(block, m, hl, p, day_lo, day_hi, cv):
    """Log markdown L[path, day] <= 0. Launch w.p. p on a uniform day, size
    M = m * lognormal(cv), decaying with half-life hl (inf = persistent)."""
    n = block["u_occ"].shape[0]
    if m == 0.0:
        return np.zeros((n, N_DAYS))
    occ = block["u_occ"] < p
    tau = day_lo + np.floor(block["u_day"] * (day_hi - day_lo + 1)).astype(int)   # 1..N
    mag = m * np.exp(cv * block["z_mag"] - 0.5 * cv * cv) if cv > 0 else np.full(n, m)
    age = np.arange(1, N_DAYS + 1)[None, :] - tau[:, None]
    active = (age >= 0) & occ[:, None]
    rho = 1.0 if math.isinf(hl) else 0.5 ** (1.0 / hl)
    return -(mag[:, None] * rho ** np.clip(age, 0, None)) * active


def settlements(blocks, m, hl, p, day_lo, day_hi, cv):
    """Per block: (final daily print, arithmetic average of the 30 prints)."""
    out = []
    for b in blocks:
        s = b["base"] * np.exp(markdown_log(b, m, hl, p, day_lo, day_hi, cv))
        out.append((s[:, -1].copy(), s.mean(axis=1)))
    return out


# ----------------------------------------------------------------------------
# Black-76 smile from a Monte Carlo sample
# ----------------------------------------------------------------------------
def _otm_norm_price(w, k):
    """Undiscounted Black-76 OTM price / forward. k = ln(K/F), w = sigma*sqrt(T)."""
    d1 = -k / w + 0.5 * w
    d2 = d1 - w
    call = ndtr(d1) - np.exp(k) * ndtr(d2)
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
    return np.where(k >= 0, call, put)


def _implied_w(price, k):
    lo = np.full_like(k, 1e-7)
    hi = np.full_like(k, 6.0)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        too_low = _otm_norm_price(mid, k) < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    w = 0.5 * (lo + hi)
    bad = (price <= 0) | (w < 2e-7) | (w > 5.99)
    return np.where(bad, np.nan, w)


def _cross(xk, yv, target):
    """Linear-interpolated crossings of yv == target along xk."""
    d = yv - target
    idx = np.where(np.isfinite(d[:-1]) & np.isfinite(d[1:]) & (d[:-1] * d[1:] <= 0) & (d[:-1] != d[1:]))[0]
    return [xk[i] + (xk[i + 1] - xk[i]) * (-d[i]) / (d[i + 1] - d[i]) for i in idx]


def smile(x, n_grid=121, width=3.0):
    """Black-76 smile for settlement sample x, forward = sample mean (own forward).

    Returns ATM vol, 25-delta risk reversal (call IV - put IV, forward deltas),
    the matched |log-moneyness| risk reversal at k = +/-0.6745*sigma_ATM*sqrt(T),
    and a reliability flag. Vols are annualised decimals.
    """
    n = x.shape[0]
    fwd = x.mean()
    xs = np.sort(x)
    cs = np.concatenate(([0.0], np.cumsum(xs)))
    sd = np.log(xs).std()
    k = sd * np.linspace(-width, width, n_grid)
    strikes = fwd * np.exp(k)
    idx = np.searchsorted(xs, strikes)
    put = (strikes * idx - cs[idx]) / n
    call = ((cs[-1] - cs[idx]) - strikes * (n - idx)) / n
    w = _implied_w(np.where(k >= 0, call, put) / fwd, k)
    iv = w / math.sqrt(T_YEARS)
    delta_call = ndtr(-k / w + 0.5 * w)          # forward call delta; put delta = this - 1
    flags = []
    res = dict(fwd=fwd, atm=float(np.interp(0.0, k, iv)))

    c25 = [c for c in _cross(k, delta_call, 0.25) if c > 0]
    p25 = [c for c in _cross(k, delta_call, 0.75) if c < 0]
    if len(c25) != 1 or len(p25) != 1:
        flags.append("delta_not_unique")
    if c25 and p25:
        kc, kp = min(c25), max(p25)
        ivc, ivp = float(np.interp(kc, k, iv)), float(np.interp(kp, k, iv))
        res.update(k25c=kc, k25p=kp, iv25c=ivc, iv25p=ivp, rr25=ivc - ivp)
    else:
        res.update(k25c=np.nan, k25p=np.nan, iv25c=np.nan, iv25p=np.nan, rr25=np.nan)
        flags.append("delta_not_bracketed")
    kstar = 0.6745 * res["atm"] * math.sqrt(T_YEARS)
    res["kstar"] = kstar
    res["rrk"] = float(np.interp(kstar, k, iv) - np.interp(-kstar, k, iv))
    if not np.isfinite(res["rr25"]) or not np.isfinite(res["rrk"]):
        flags.append("nan_iv")
    res["flags"] = flags
    return res


def scenario_stats(setts):
    """Pooled estimate + block standard errors for both settlements."""
    out = {}
    for j, name in enumerate(("fin", "avg")):
        pooled = smile(np.concatenate([s[j] for s in setts]))
        blk = [smile(s[j]) for s in setts]
        r = len(blk)
        for key in ("rr25", "rrk", "atm"):
            vals = np.array([b[key] for b in blk], dtype=float)
            out[f"{key}_{name}"] = pooled[key]
            out[f"{key}_{name}_se"] = float(np.nanstd(vals, ddof=1) / math.sqrt(r)) if r > 1 else float("nan")
        flags = list(pooled["flags"])
        if any(b["flags"] for b in blk):
            flags.append("block_flag")
        if np.isfinite(pooled["rr25"]) and np.isfinite(pooled["rrk"]) and pooled["rr25"] * pooled["rrk"] < 0 \
                and abs(pooled["rr25"]) > Z_CRIT * out[f"rr25_{name}_se"]:
            flags.append("moneyness_crosscheck_disagrees")
        out[f"flags_{name}"] = sorted(set(flags))
        out[f"fwd_{name}"] = pooled["fwd"]
        out[f"k25c_{name}"], out[f"k25p_{name}"] = pooled["k25c"], pooled["k25p"]
    z = Z_CRIT
    out["flip"] = bool(
        not out["flags_fin"] and not out["flags_avg"]
        and out["rr25_fin"] - z * out["rr25_fin_se"] > 0 and out["rr25_avg"] + z * out["rr25_avg_se"] < 0
        and out["rrk_fin"] - z * out["rrk_fin_se"] > 0 and out["rrk_avg"] + z * out["rrk_avg_se"] < 0)
    out["both_put"] = bool(out["rr25_fin"] + z * out["rr25_fin_se"] < 0 and out["rr25_avg"] + z * out["rr25_avg_se"] < 0)
    return out


def run_grid(blocks, par, sizes, halflives):
    rows = []
    for hl in halflives:
        for m in sizes:
            st = scenario_stats(settlements(blocks, m, hl, par["p"], par["day_lo"], par["day_hi"], par["mag_cv"]))
            st.update(m=m, hl=hl)
            rows.append(st)
    return rows


def threshold(rows):
    """Smallest tested markdown with a credible flip, per half-life and overall."""
    per = {}
    for r in rows:
        if r["flip"] and (r["hl"] not in per or r["m"] < per[r["hl"]]):
            per[r["hl"]] = r["m"]
    return per, (min(per.values()) if per else None)


# ----------------------------------------------------------------------------
# Ornn data
# ----------------------------------------------------------------------------
def ornn_stats(refresh=False):
    info = dict(endpoint=ORNN_URL, raw_file=os.path.relpath(ORNN_RAW, HERE), available=False)
    if refresh or not os.path.exists(ORNN_RAW):
        try:
            os.makedirs(os.path.dirname(ORNN_RAW), exist_ok=True)
            req = urllib.request.Request(ORNN_URL, headers={"User-Agent": "research-client/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
            with open(ORNN_RAW, "wb") as f:
                f.write(raw)
        except Exception as exc:                       # proceed with illustrative inputs
            info["error"] = repr(exc)
            return info
    with open(ORNN_RAW, "r", encoding="utf-8") as f:
        payload = json.load(f)
    recs = sorted(payload["data"], key=lambda r: r["timestamp"])
    dates = [np.datetime64(r["timestamp"][:10]) for r in recs]
    v = np.array([float(r["index_value"]) for r in recs])
    gaps = [(str(a), str(b)) for a, b in zip(dates[:-1], dates[1:]) if (b - a).astype(int) != 1]
    lv = np.log(v)
    info.update(available=True, access=payload.get("access"), gpu=payload.get("gpu_type"),
                n_obs=int(len(v)), first_date=str(dates[0]), last_date=str(dates[-1]),
                calendar_gaps=gaps, duplicates=int(len(dates) - len(set(dates))),
                level_min=float(v.min()), level_max=float(v.max()), level_last=float(v[-1]),
                level_mean=float(v.mean()))
    d1 = np.diff(lv)
    info["median_abs_1d_log"] = float(np.median(np.abs(d1)))
    info["median_abs_1d_usd"] = float(np.median(np.abs(np.diff(v))))
    info["sd_1d_log"] = float(d1.std(ddof=1))
    info["lag1_autocorr_1d_log"] = float(np.corrcoef(d1[:-1], d1[1:])[0, 1])
    info["skew_1d_log"] = float(skew(d1))
    moves = {}
    for h in (1, 5, 20):
        d = lv[h:] - lv[:-h]
        du = v[h:] - v[:-h]
        moves[str(h)] = dict(n=int(len(d)), max_up_log=float(d.max()), max_down_log=float(d.min()),
                             max_up_pct=float(100 * (math.exp(d.max()) - 1)),
                             max_down_pct=float(100 * (math.exp(d.min()) - 1)),
                             max_up_usd=float(du.max()), max_down_usd=float(du.min()),
                             sd_log=float(d.std(ddof=1)))
    info["moves"] = moves
    return info


def model_move_stats(blocks):
    """Same statistics for the launch-free simulated process (context for scale)."""
    lb = np.log(np.concatenate([b["base"] for b in blocks]))
    out = {}
    d1 = np.diff(lb, axis=1).ravel()
    out["median_abs_1d_log"] = float(np.median(np.abs(d1)))
    out["sd_1d_log"] = float(d1.std())
    for h in (1, 5, 20):
        d = (lb[:, h:] - lb[:, :-h]).ravel()
        out[f"sd_{h}d_log"] = float(d.std())
        out[f"q001_{h}d_log"], out[f"q999_{h}d_log"] = [float(q) for q in np.quantile(d, [0.001, 0.999])]
    return out


# ----------------------------------------------------------------------------
# Controls and checks
# ----------------------------------------------------------------------------
def daily_return_skew(blocks, par, m, hl):
    """Skewness of daily log returns ln(S_t/S_{t-1}), t = 1..30 with S_0 fixed.
    'all': every day of every path pooled. 'ex_launch_day': same, dropping the
    single launch-day return on launch paths. 'no_launch_paths': paths without a launch."""
    allr, exr, nol = [], [], []
    for b in blocks:
        logs = np.log(b["base"]) + markdown_log(b, m, hl, par["p"], par["day_lo"], par["day_hi"], par["mag_cv"])
        full = np.concatenate([np.full((logs.shape[0], 1), math.log(par["S0"])), logs], axis=1)
        ret = np.diff(full, axis=1)
        occ = b["u_occ"] < par["p"] if m > 0 else np.zeros(logs.shape[0], bool)
        tau = par["day_lo"] + np.floor(b["u_day"] * (par["day_hi"] - par["day_lo"] + 1)).astype(int)
        keep = np.ones_like(ret, bool)
        keep[np.where(occ)[0], tau[occ] - 1] = False
        allr.append(ret.ravel()); exr.append(ret[keep]); nol.append(ret[~occ].ravel())
    f = lambda parts: float(skew(np.concatenate(parts)))
    return dict(all=f(allr), ex_launch_day=f(exr), no_launch_paths=f(nol),
                n_returns_all=int(sum(a.size for a in allr)))


def additive_check(blocks, J=-0.60, d=10):
    """Additive shock J present for d of N days moves the arithmetic average by J*d/N."""
    base = blocks[0]["base"]
    res = {}
    for label, start in (("days_6_to_15", 5), ("days_21_to_30", N_DAYS - d)):
        s = base.copy()
        s[:, start:start + d] += J
        res[label] = dict(
            avg_change_max_abs_error=float(np.abs((s.mean(1) - base.mean(1)) - J * d / N_DAYS).max()),
            avg_change=float((s.mean(1) - base.mean(1)).mean()), predicted=J * d / N_DAYS,
            final_change=float((s[:, -1] - base[:, -1]).mean()))
    return res


def diffusion_benchmark(n, seed):
    """Random-walk only: final print is lognormal, so Black-76 IV must be flat at
    rw_sd*sqrt(365) and the risk reversal must be zero."""
    par = dict(BASE, ou_sd=0.0, spk_lam=0.0, rw_sd=0.02)
    blocks = [build_block(seed + i, n, par) for i in range(8)]
    st = scenario_stats(settlements(blocks, 0.0, math.inf, 0.5, 1, 30, 0.0))
    return dict(analytic_iv=0.02 * math.sqrt(365), mc_atm_iv=st["atm_fin"], mc_atm_se=st["atm_fin_se"],
                mc_rr25=st["rr25_fin"], mc_rr25_se=st["rr25_fin_se"])


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def hl_label(h):
    return "persists" if math.isinf(h) else f"{h:g}d"


def make_figure(rows, sizes, halflives, rep_hl, path):
    nx, ny = len(sizes), len(halflives)
    cell = {(r["hl"], r["m"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    for ax, key, title in ((axes[0], "fin", "Final daily print: 25Δ risk reversal"),
                           (axes[1], "avg", "Monthly average: 25Δ risk reversal")):
        z = np.array([[100 * cell[(h, m)][f"rr25_{key}"] for m in sizes] for h in halflives])
        vmax = max(1.0, float(np.nanpercentile(np.abs(z), 90)))
        im = ax.imshow(z, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect="auto")
        for i, h in enumerate(halflives):
            for j, m in enumerate(sizes):
                c = cell[(h, m)]
                val, se = 100 * c[f"rr25_{key}"], 100 * c[f"rr25_{key}_se"]
                weak = abs(val) <= Z_CRIT * se
                txt = "n/a" if not np.isfinite(val) else f"{val:+.1f}"
                ax.text(j, i, txt + ("\n(ns)" if weak else ""), ha="center", va="center", fontsize=7.5,
                        color="white" if abs(val) > 0.6 * vmax else "black")
                if c[f"flags_{key}"]:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", ec="0.35", lw=0))
                if c["flip"]:
                    ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, ec="black", lw=2.2))
        ax.set_xticks(range(nx), [f"{m:g}\n({100 * (1 - math.exp(-m)):.0f}%)" for m in sizes], fontsize=8)
        ax.set_yticks(range(ny), [hl_label(h) for h in halflives])
        ax.set_xlabel("mean launch markdown, log units (and % price cut)")
        ax.set_ylabel("markdown half-life (recovery speed)")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="call IV − put IV, vol points")
    ax = axes[2]
    xs = np.array(sizes)
    for key, col, lab in (("fin", "#b2182b", "final daily print"), ("avg", "#2166ac", "monthly average")):
        y = np.array([100 * cell[(rep_hl, m)][f"rr25_{key}"] for m in sizes])
        e = np.array([100 * Z_CRIT * cell[(rep_hl, m)][f"rr25_{key}_se"] for m in sizes])
        ax.errorbar(xs, y, yerr=e, color=col, marker="o", ms=4, capsize=3, lw=1.6, label=f"{lab} (25Δ, ±{Z_CRIT:g} s.e.)")
        yk = np.array([100 * cell[(rep_hl, m)][f"rrk_{key}"] for m in sizes])
        ax.plot(xs, yk, color=col, ls=":", lw=1.2, label=f"{lab} (matched |log-moneyness|)")
    flips = [m for m in sizes if cell[(rep_hl, m)]["flip"]]
    if flips:
        ax.axvspan(min(flips) - 0.006, max(flips) + 0.006, color="0.85", zorder=0, label="credible sign flip")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("mean launch markdown, log units")
    ax.set_ylabel("risk reversal, vol points")
    ax.set_title(f"Slice at half-life {hl_label(rep_hl)}", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Same simulated H100 price paths, two hypothetical 30-day options (assumed pricing measure, illustrative parameters).\n"
                 "Black boxes: final-print RR > 0 and average RR < 0, each beyond "
                 f"{Z_CRIT:g} Monte Carlo s.e. on both the 25Δ and matched-moneyness measures.  "
                 "Hatching: unreliable 25Δ estimate.  (ns): within simulation error of zero.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh-ornn", action="store_true", help="re-download the Ornn history")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n_blk, n_per = (8, 10_000) if args.quick else (16, 25_000)
    s_blk, s_per = (8, 5_000) if args.quick else (8, 20_000)
    v_blk, v_per = (8, 20_000) if args.quick else (16, 100_000)
    summary = dict(seed=SEED, n_days=N_DAYS, T_years=T_YEARS, z_crit=Z_CRIT, baseline=BASE,
                   sizes=SIZES, halflives=[hl_label(h) for h in HALFLIVES],
                   main_paths=n_blk * n_per, main_blocks=n_blk)

    # 1. Ornn data --------------------------------------------------------------
    ornn = ornn_stats(args.refresh_ornn)
    summary["ornn"] = ornn
    print("Ornn:", json.dumps(clean(ornn), indent=1)[:1800])

    # 2. Main sweep ---------------------------------------------------------------
    blocks = [build_block(SEED + i, n_per, BASE) for i in range(n_blk)]
    summary["model_moves_no_launch"] = model_move_stats(blocks)
    print("model moves:", summary["model_moves_no_launch"])
    rows = run_grid(blocks, BASE, SIZES, HALFLIVES)
    per, m_star = threshold(rows)
    print(f"[{time.time() - t0:5.1f}s] main sweep done; threshold per half-life:",
          {hl_label(k): v for k, v in per.items()}, "overall", m_star)
    hdr = "   hl " + "".join(f"{m:>17g}" for m in SIZES)
    for key in ("fin", "avg"):
        print(f"RR25 {key} (vol pts, * = credible flip cell)\n" + hdr)
        for h in HALFLIVES:
            print(f"{hl_label(h):>6}" + "".join(
                f"{100 * r[f'rr25_{key}']:+8.2f}+/-{100 * r[f'rr25_{key}_se']:4.2f}{'*' if r['flip'] else ' '}"
                for r in rows if r["hl"] == h))
    cols = ["hl", "m", "flip", "both_put"] + [f"{a}_{b}{c}" for b in ("fin", "avg")
                                              for a in ("rr25", "rrk", "atm") for c in ("", "_se")] + \
           ["fwd_fin", "fwd_avg", "k25c_fin", "k25p_fin", "k25c_avg", "k25p_avg", "flags_fin", "flags_avg"]
    with open(os.path.join(OUT, "grid_results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([hl_label(r[c]) if c == "hl" else ("|".join(r[c]) if c.startswith("flags") else r[c]) for c in cols])

    # representative cell: at the threshold size, the half-life with the widest margin
    cell = {(r["hl"], r["m"]): r for r in rows}
    if m_star is not None:
        cands = [r for r in rows if r["flip"] and r["m"] == m_star]
        rep = max(cands, key=lambda r: min(r["rr25_fin"] / r["rr25_fin_se"], -r["rr25_avg"] / r["rr25_avg_se"]))
        rep_hl = rep["hl"]
    else:
        rep, rep_hl = None, 5.0
    make_figure(rows, SIZES, HALFLIVES, rep_hl, os.path.join(OUT, "skew_main_figure.png"))
    print(f"[{time.time() - t0:5.1f}s] figure saved")
    summary["threshold_per_halflife"] = {hl_label(k): v for k, v in per.items()}
    summary["threshold_overall"] = m_star
    summary["flip_cells"] = [dict(hl=hl_label(r["hl"]), m=r["m"]) for r in rows if r["flip"]]
    summary["both_put_cells"] = [dict(hl=hl_label(r["hl"]), m=r["m"]) for r in rows if r["both_put"]]
    summary["flagged_cells"] = [dict(hl=hl_label(r["hl"]), m=r["m"], fin=r["flags_fin"], avg=r["flags_avg"])
                                for r in rows if r["flags_fin"] or r["flags_avg"]]
    keep = ("rr25_fin", "rr25_fin_se", "rr25_avg", "rr25_avg_se", "rrk_fin", "rrk_fin_se", "rrk_avg",
            "rrk_avg_se", "atm_fin", "atm_avg", "fwd_fin", "fwd_avg", "k25c_fin", "k25p_fin", "k25c_avg",
            "k25p_avg", "flip")
    summary["no_launch_control"] = {k: cell[(HALFLIVES[0], 0.0)][k] for k in keep}
    if rep is not None:
        summary["representative"] = dict(hl=hl_label(rep_hl), m=m_star, **{k: rep[k] for k in keep})
        below = [m for m in SIZES if m < m_star]
        if below:
            summary["cell_below_threshold"] = dict(hl=hl_label(rep_hl), m=below[-1],
                                                   **{k: cell[(rep_hl, below[-1])][k] for k in keep})

    # 3. Controls -------------------------------------------------------------------
    det = scenario_stats(settlements(blocks, 0.20, rep_hl, 1.0, 10, 10, 0.0))
    det_p = scenario_stats(settlements(blocks, 0.20, math.inf, 1.0, 10, 10, 0.0))
    summary["deterministic_launch_control"] = dict(
        note="launch certain, day 10, size 0.20 exactly",
        recovering={k: det[k] for k in keep}, persistent={k: det_p[k] for k in keep})
    summary["additive_shock_check"] = additive_check(blocks)
    summary["diffusion_only_benchmark"] = diffusion_benchmark(n_per, SEED + 5000)
    skew_cases = {"no_launch": (0.0, math.inf), "persistent_m0.20": (0.20, math.inf)}
    if rep is not None:
        skew_cases["representative"] = (m_star, rep_hl)
    summary["daily_return_skew"] = {k: daily_return_skew(blocks, BASE, *v) for k, v in skew_cases.items()}
    print("controls:", json.dumps(clean({k: summary[k] for k in (
        "no_launch_control", "deterministic_launch_control", "additive_shock_check",
        "diffusion_only_benchmark", "daily_return_skew")}), indent=1))

    # 4. Independent verification of the representative flip ------------------------
    if rep is not None:
        ver = {}
        vblocks_seed = SEED + 100_000
        for label, m in (("threshold", m_star),) + ((("below", below[-1]),) if below else ()):
            setts = []
            for i in range(v_blk):
                b = build_block(vblocks_seed + i, v_per, BASE)
                s = b["base"] * np.exp(markdown_log(b, m, rep_hl, BASE["p"], BASE["day_lo"], BASE["day_hi"], BASE["mag_cv"]))
                setts.append((s[:, -1].copy(), s.mean(axis=1)))
            st = scenario_stats(setts)
            ver[label] = dict(m=m, hl=hl_label(rep_hl), paths=v_blk * v_per, **{k: st[k] for k in keep})
        summary["independent_verification"] = ver
        print(f"[{time.time() - t0:5.1f}s] verification:", json.dumps(clean(ver), indent=1))

    # 5. Sensitivity (common random numbers; spike changes rebuild paths from the same seeds)
    sens_specs = [
        ("baseline", {}),
        ("timing: early (days 1-10)", dict(day_lo=1, day_hi=10)),
        ("timing: mid (days 11-20)", dict(day_lo=11, day_hi=20)),
        ("timing: late (days 21-30)", dict(day_lo=21, day_hi=30)),
        ("timing: known (day 15)", dict(day_lo=15, day_hi=15)),
        ("launch prob 0.10", dict(p=0.10)),
        ("launch prob 0.25", dict(p=0.25)),
        ("launch prob 0.75", dict(p=0.75)),
        ("launch prob 1.00", dict(p=1.00)),
        ("magnitude known (cv=0)", dict(mag_cv=0.0)),
        ("spike half-life 1d", dict(spk_hl=1.0)),
        ("spike half-life 7d", dict(spk_hl=7.0)),
        ("spikes 0.5/yr, half-life 7d (Lee data-consistent)", dict(spk_lam=0.5, spk_hl=7.0)),
        ("no spikes", dict(spk_lam=0.0)),
    ]
    sens_rows, cache = [], {}
    for name, over in sens_specs:
        par = dict(BASE, **over)
        ck = (par["spk_lam"], par["spk_hl"])
        if ck not in cache:
            cache[ck] = [build_block(SEED + i, s_per, par) for i in range(s_blk)]
        r = run_grid(cache[ck], par, SIZES, SENS_HALFLIVES)
        p_, ms = threshold(r)
        c0 = [x for x in r if x["m"] == 0.0][0]
        sens_rows.append(dict(name=name, threshold_overall=ms,
                              per_halflife={hl_label(h): p_.get(h) for h in SENS_HALFLIVES},
                              n_flip_cells=sum(x["flip"] for x in r), n_both_put=sum(x["both_put"] for x in r),
                              rr25_fin_nolaunch=c0["rr25_fin"], rr25_avg_nolaunch=c0["rr25_avg"]))
        print(f"[{time.time() - t0:5.1f}s] {name:<50} threshold {ms}  per-hl {sens_rows[-1]['per_halflife']}")
    summary["sensitivity"] = sens_rows
    summary["sensitivity_paths"] = s_blk * s_per
    with open(os.path.join(OUT, "sensitivity_results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["setting", "threshold_overall"] + [f"threshold_hl_{hl_label(h)}" for h in SENS_HALFLIVES] +
                    ["n_flip_cells", "n_both_put_cells", "rr25_fin_nolaunch", "rr25_avg_nolaunch"])
        for s in sens_rows:
            wr.writerow([s["name"], s["threshold_overall"]] + [s["per_halflife"][hl_label(h)] for h in SENS_HALFLIVES] +
                        [s["n_flip_cells"], s["n_both_put"], s["rr25_fin_nolaunch"], s["rr25_avg_nolaunch"]])

    # 6. Threshold vs observed moves ---------------------------------------------------
    if rep is not None and ornn.get("available"):
        rho = 1.0 if math.isinf(rep_hl) else 0.5 ** (1.0 / rep_hl)
        d_eff = sum(rho ** j for j in range(N_DAYS))    # launch on day 1
        comp = dict(threshold_log=m_star, threshold_pct=100 * (math.exp(-m_star) - 1),
                    threshold_usd_at_last_level=ornn["level_last"] * (math.exp(-m_star) - 1),
                    ratio_to_median_abs_1d=m_star / ornn["median_abs_1d_log"],
                    ratio_to_largest_1d_down=m_star / abs(ornn["moves"]["1"]["max_down_log"]),
                    residual_after_5_prints_log=-m_star * rho ** 4,
                    residual_after_20_prints_log=-m_star * rho ** 19,
                    ornn_largest_5d_down_log=ornn["moves"]["5"]["max_down_log"],
                    ornn_largest_20d_down_log=ornn["moves"]["20"]["max_down_log"],
                    max_effective_days_in_average=d_eff,
                    max_log_dent_in_monthly_average=m_star * d_eff / N_DAYS)
        summary["threshold_vs_ornn"] = comp
        print("threshold vs Ornn:", json.dumps(clean(comp), indent=1))

    summary["runtime_seconds"] = time.time() - t0
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(clean(summary), f, indent=1)
    print(f"done in {time.time() - t0:.1f}s; outputs in {OUT}")


if __name__ == "__main__":
    main()
