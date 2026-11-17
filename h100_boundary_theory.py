#!/usr/bin/env python
"""Closed-form rule for the skew sign boundary, checked against the simulations.

    python h100_boundary_theory.py        (about 2-3 min, CPU, fixed seed)

Run h100_skew_demo.py and h100_conditional_demo.py first: this script reads
outputs/grid_results.csv and outputs/conditional_repricing.json for comparison.

Two layers, both without Monte Carlo:

1. Third-cumulant balance (closed form). The log settlement is a sum of
   independent pieces: Gaussian noise (no skew), upward spikes (positive third
   cumulant) and the launch markdown (third cumulant = -c * m^3). Skewness is
   zero at
        m* = (kappa3_spike / c) ** (1/3)
   where kappa3_spike and c depend only on how long each shock stays in the
   settlement (its loading) and on the launch probability.

2. Semi-analytic smile. The characteristic function of the log settlement is a
   product of closed-form factors, so the density comes from one FFT and the
   25-delta risk reversal follows from Black-76. Exact for the final print;
   for the average each single shock's effect on the log arithmetic average is
   exact and only interactions between shocks are approximated.

Outputs: outputs/boundary_theory_figure.png, outputs/boundary_theory.json
"""
import csv
import json
import math
import os
import time

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import skew as sample_skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h100_skew_demo import (BASE, SEED, N_DAYS, OUT, Z_CRIT, build_block, settlements,
                            scenario_stats, _implied_w, _cross, clean)

INF = math.inf


# ----------------------------------------------------------------------------
# Loadings: how much of a shock arriving with j days still to run reaches the settlement
# ----------------------------------------------------------------------------
def _geo(ph, j):
    return (j + 1.0) if ph == 1.0 else (1.0 - ph ** (j + 1)) / (1.0 - ph)


def loadings(kind, par, hl, t=0):
    """Valuation after the day-t print (t = 0 is the start of the month).
    Returns Gaussian variance, spike loadings (one per remaining day) and launch
    loadings (one per possible launch day)."""
    N = N_DAYS
    n = N - t
    j = np.arange(n)                                   # days remaining after the shock day
    phx, phy = 0.5 ** (1 / par["ou_hl"]), 0.5 ** (1 / par["spk_hl"])
    rho = 1.0 if math.isinf(hl) else 0.5 ** (1 / hl)
    jl = N - np.arange(max(par["day_lo"], t + 1), par["day_hi"] + 1)
    if kind == "fin":
        ax, ar, ay, cl = phx ** j, np.ones(n), phy ** j, rho ** jl
    else:
        ax, ar, ay, cl = _geo(phx, j) / N, (j + 1.0) / N, _geo(phy, j) / N, _geo(rho, jl) / N
    var_g = par["ou_sd"] ** 2 * np.sum(ax ** 2) + par["rw_sd"] ** 2 * np.sum(ar ** 2)
    return var_g, ay, cl


def p_updated(p, t):
    """P(launch in days t+1..N | none by day t), launch day uniform on 1..N."""
    return p * (N_DAYS - t) / N_DAYS / (1.0 - p * t / N_DAYS)


# ----------------------------------------------------------------------------
# Layer 1: closed-form cumulant balance
# ----------------------------------------------------------------------------
def cumulant_rule(kind, hl, p, par, t=0, m=None):
    var_g, ay, cl = loadings(kind, par, hl, t)
    q, mu, s = par["spk_lam"] / 365.0, par["spk_mean"], par["mag_cv"]
    k2_spike = (2 * q * mu ** 2 - (q * mu) ** 2) * np.sum(ay ** 2)
    k3_spike = q * mu ** 3 * (6 - 6 * q + 2 * q * q) * np.sum(ay ** 3)
    a1, a2, a3 = (float(np.mean(cl ** i)) for i in (1, 2, 3))
    m1, m2, m3 = p * a1, p * math.exp(s * s) * a2, p * math.exp(3 * s * s) * a3
    c3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3              # launch third cumulant per m^3 (downward)
    c2 = m2 - m1 ** 2
    if c3 <= 0:
        mstar = INF                                   # launch never adds downside skew
    elif k3_spike <= 0:
        mstar = 0.0                                   # nothing to offset the launch
    else:
        mstar = (k3_spike / c3) ** (1.0 / 3.0)
    out = dict(mstar=mstar, k3_spike=k3_spike, c3=c3, var_g=var_g)
    if m is not None:
        out["skew"] = (k3_spike - m ** 3 * c3) / (var_g + k2_spike + m * m * c2) ** 1.5
    return out


# ----------------------------------------------------------------------------
# Layer 2: semi-analytic smile from the characteristic function
# ----------------------------------------------------------------------------
_GH_X, _GH_W = hermegauss(64)
_GH_W = _GH_W / math.sqrt(2 * math.pi)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)
_EXP_X, _EXP_W = -np.log((_GL_X + 1) / 2), _GL_W / 2      # nodes/weights for a unit exponential
M_MAX = 0.6
_CACHE = {}


def _setup(kind, par, t, n_fft):
    """FFT grid and spike factor of the characteristic function. Independent of the
    launch, so it is cached per (settlement, valuation day, spot parameters)."""
    key = (kind, t, n_fft, par["ou_sd"], par["ou_hl"], par["rw_sd"], par["spk_lam"], par["spk_mean"], par["spk_hl"])
    if key in _CACHE:
        return _CACHE[key]
    N, n = N_DAYS, N_DAYS - t
    var_g, ay, _ = loadings(kind, par, INF, t)
    q, mu = par["spk_lam"] / 365.0, par["spk_mean"]
    phy = 0.5 ** (1 / par["spk_hl"])
    sg = math.sqrt(var_g)
    if kind == "fin":
        spike_max = 25 * mu
    else:
        spike_max = math.log1p(np.sum(np.expm1(20 * mu * phy ** np.arange(n))) / N)
    zmin = -(10 * sg + 5 * M_MAX * (1.0 if kind == "fin" else n / N) + 0.02)
    zmax = 10 * sg + (spike_max if q > 0 else 0.0) + 0.02
    dz = (zmax - zmin) / n_fft
    du = 2 * math.pi / (n_fft * dz)
    u = du * np.arange(n_fft)
    act = u < math.sqrt(90.0 / var_g)
    ua = u[act]
    ph = np.exp(-0.5 * var_g * ua ** 2).astype(complex)
    if q > 0:
        if kind == "fin":                      # log effect of a spike is linear: closed form
            for a in ay:
                ph *= (1 - q) + q / (1 - 1j * ua * mu * a)
        else:                                  # exact log effect of one spike on the arithmetic average
            jump = mu * _EXP_X
            eff = np.log1p(np.cumsum(np.expm1(jump[None, :] * (phy ** np.arange(n))[:, None]), axis=0) / N)
            for j in range(n):
                ph *= (1 - q) + q * (np.exp(1j * ua[:, None] * eff[j][None, :]) * _EXP_W).sum(axis=1)
    out = dict(var_g=var_g, zmin=zmin, dz=dz, u=u, act=act, ua=ua, ph=ph, z=zmin + dz * np.arange(n_fft))
    _CACHE[key] = out
    return out


def launch_effects(kind, par, hl, t, mv):
    """Log effect on the settlement of a launch of size mv[node] on each possible launch day."""
    N = N_DAYS
    rho = 1.0 if math.isinf(hl) else 0.5 ** (1 / hl)
    jl = N - np.arange(max(par["day_lo"], t + 1), par["day_hi"] + 1)
    if kind == "fin":
        return -(rho ** jl)[:, None] * mv[None, :]
    k = np.arange(jl.max() + 1)
    cum = np.cumsum(-np.expm1(-mv[None, :] * (rho ** k)[:, None]), axis=0) / N      # (k, node)
    return np.log1p(-cum[jl])


def smile_theory(kind, m, hl, p, par, t=0, n_fft=2 ** 14, n_k=81):
    g = _setup(kind, par, t, n_fft)
    n = N_DAYS - t
    s = par["mag_cv"]
    ua, z, dz = g["ua"], g["z"], g["dz"]
    ph = g["ph"]
    if m > 0 and p > 0:
        mv, mw = (m * np.exp(s * _GH_X - 0.5 * s * s), _GH_W) if s > 0 else (np.array([m]), np.array([1.0]))
        eff = launch_effects(kind, par, hl, t, mv)                                   # (launch day, node)
        e = np.exp(1j * ua[:, None, None] * eff[None, :, :])
        ph = ph * ((1 - p) + p * (e * mw).sum(axis=2).mean(axis=1))
    phi = np.zeros(n_fft, complex)
    phi[g["act"]] = ph
    phi[0] *= 0.5
    du = 2 * math.pi / (n_fft * dz)
    f = (du / math.pi) * np.real(np.fft.fft(phi * np.exp(-1j * g["u"] * g["zmin"])))
    f = np.clip(f, 0, None)
    f /= f.sum() * dz
    x = np.exp(z - math.log(np.sum(f * np.exp(z)) * dz))      # settlement / its own forward
    mean = np.sum(f * z) * dz
    sd = math.sqrt(np.sum(f * (z - mean) ** 2) * dz)
    k = sd * np.linspace(-3, 3, n_k)
    kk = np.exp(k)[:, None]
    call = (np.clip(x[None, :] - kk, 0, None) * f).sum(axis=1) * dz
    put = (np.clip(kk - x[None, :], 0, None) * f).sum(axis=1) * dz
    w = _implied_w(np.where(k >= 0, call, put), k)
    delta = ndtr(-k / w + 0.5 * w)
    ann = math.sqrt(n / 365.0)
    out = dict(atm=float(np.interp(0.0, k, w)) / ann,
               skew=float(np.sum(f * (z - mean) ** 3) * dz / sd ** 3))
    for d, name in ((0.25, "rr25"), (0.10, "rr10")):
        kc = [c for c in _cross(k, delta, d) if c > 0]
        kp = [c for c in _cross(k, delta, 1 - d) if c < 0]
        out[name] = (float(np.interp(min(kc), k, w) - np.interp(max(kp), k, w)) / ann) if kc and kp else float("nan")
    return out


def threshold(fn, mmax=0.6, step=0.03):
    """First m where fn(m) turns negative. 0.0 if negative from the start, inf if never."""
    if fn(1e-4) <= 0:
        return 0.0
    prev = 1e-4
    for mm in np.arange(step, mmax + 1e-9, step):
        if fn(mm) < 0:
            return float(brentq(fn, prev, mm, xtol=5e-4))
        prev = mm
    return INF


def rr_threshold(kind, hl, p, par, t=0, key="rr25"):
    return threshold(lambda mm: smile_theory(kind, mm, hl, p, par, t)[key])


# ----------------------------------------------------------------------------
def hl_of(label):
    return INF if label == "persists" else float(label.rstrip("d"))


def hl_label(h):
    return "persists" if math.isinf(h) else f"{h:g}d"


def mc_crossing(ms, vals):
    for i in range(len(ms) - 1):
        if vals[i] > 0 and vals[i + 1] <= 0:
            return ms[i] + (ms[i + 1] - ms[i]) * vals[i] / (vals[i] - vals[i + 1])
    return INF


def main():
    t0 = time.time()
    par = dict(BASE)
    p = par["p"]
    res = {}

    # ---- A. pricer vs the Monte Carlo grid ---------------------------------------
    with open(os.path.join(OUT, "grid_results.csv")) as f:
        grid = list(csv.DictReader(f))
    errs = {"fin": [], "avg": []}
    worst = {"fin": None, "avg": None}
    for r in grid:
        h, m = hl_of(r["hl"]), float(r["m"])
        for kind in ("fin", "avg"):
            th = smile_theory(kind, m, h, p, par)["rr25"]
            mc, se = float(r[f"rr25_{kind}"]), float(r[f"rr25_{kind}_se"])
            e = 100 * (th - mc)
            errs[kind].append(e)
            if worst[kind] is None or abs(e) > abs(worst[kind]["err"]):
                worst[kind] = dict(hl=r["hl"], m=m, theory=100 * th, mc=100 * mc, mc_se=100 * se, err=e)
    res["pricer_vs_mc_grid"] = {k: dict(n=len(v), median_abs_err=float(np.median(np.abs(v))),
                                        max_abs_err=float(np.max(np.abs(v))), worst=worst[k]) for k, v in errs.items()}
    print(f"[{time.time() - t0:5.1f}s] pricer vs MC grid (vol pts):", json.dumps(clean(res["pricer_vs_mc_grid"]), indent=1))

    # ---- B. boundaries by half-life: pricer, cumulant rule, Monte Carlo -----------
    hls_mc = sorted({hl_of(r["hl"]) for r in grid})
    hls_th = [1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 60, INF]
    bound = []
    for h in hls_th:
        row = dict(hl=hl_label(h))
        for kind in ("fin", "avg"):
            row[f"rr25_{kind}"] = rr_threshold(kind, h, p, par)
            row[f"rr10_{kind}"] = rr_threshold(kind, h, p, par, key="rr10")
            row[f"rule_{kind}"] = cumulant_rule(kind, h, p, par)["mstar"]
            row[f"skew_exact_{kind}"] = threshold(lambda mm: smile_theory(kind, mm, h, p, par)["skew"])
            if h in hls_mc:
                rows = sorted([r for r in grid if hl_of(r["hl"]) == h], key=lambda r: float(r["m"]))
                row[f"mc_{kind}"] = mc_crossing([float(r["m"]) for r in rows], [float(r[f"rr25_{kind}"]) for r in rows])
        bound.append(row)
        print(f"[{time.time() - t0:5.1f}s] hl {row['hl']:>8}: " + "  ".join(
            f"{kind}: RR25 {row[f'rr25_{kind}']:.3f} (MC {row.get(f'mc_{kind}', float('nan')):.3f}) "
            f"RR10 {row[f'rr10_{kind}']:.3f} rule {row[f'rule_{kind}']:.3f} skew0 {row[f'skew_exact_{kind}']:.3f}"
            for kind in ("avg", "fin")))
    res["boundaries"] = bound

    # ---- C. Monte Carlo skewness zero vs the rule (tests the rule itself) ----------
    blocks = [build_block(SEED + i, 25_000, par) for i in range(8)]
    ms = np.round(np.arange(0.02, 0.321, 0.02), 3)
    skew_check = []
    for h in (10.0, INF):
        sk = {"fin": [], "avg": []}
        for mm in ms:
            st = settlements(blocks, float(mm), h, p, par["day_lo"], par["day_hi"], par["mag_cv"])
            sk["fin"].append(float(sample_skew(np.log(np.concatenate([s[0] for s in st])))))
            sk["avg"].append(float(sample_skew(np.log(np.concatenate([s[1] for s in st])))))
        skew_check.append(dict(hl=hl_label(h), **{f"mc_skew_zero_{k}": mc_crossing(list(ms), v) for k, v in sk.items()},
                               **{f"rule_{k}": cumulant_rule(k, h, p, par)["mstar"] for k in ("fin", "avg")}))
    res["skewness_zero_mc_vs_rule"] = skew_check
    print(f"[{time.time() - t0:5.1f}s] skewness zero, MC vs rule:", json.dumps(clean(skew_check)))

    # ---- D. scaling collapse: rule prediction vs pricer threshold ------------------
    coll = []
    for lam in (3.0, 6.0, 12.0, 24.0):
        for mu in (0.10, 0.15, 0.20):
            for shl in (1.0, 2.0, 4.0):
                pp = dict(par, spk_lam=lam, spk_mean=mu, spk_hl=shl)
                coll.append(dict(lam=lam, mu=mu, spk_hl=shl, p=p,
                                 rule_avg=cumulant_rule("avg", INF, p, pp)["mstar"], rr25_avg=rr_threshold("avg", INF, p, pp),
                                 rule_fin=cumulant_rule("fin", INF, p, pp)["mstar"], rr25_fin=rr_threshold("fin", INF, p, pp)))
    res["collapse"] = coll
    ra = np.array([c["rr25_avg"] / c["rule_avg"] for c in coll if math.isfinite(c["rr25_avg"])])
    rf = np.array([c["rr25_fin"] / c["rule_fin"] for c in coll if math.isfinite(c["rr25_fin"])])
    res["collapse_ratio"] = dict(avg_min=float(ra.min()), avg_max=float(ra.max()), avg_median=float(np.median(ra)),
                                 fin_min=float(rf.min()), fin_max=float(rf.max()), fin_median=float(np.median(rf)))
    print(f"[{time.time() - t0:5.1f}s] RR25 threshold / rule:", res["collapse_ratio"])

    # ---- D2. the flip band under other spike settings (persistent markdown) ----------
    alt = []
    for name, pp in (("baseline (12/yr, 2d)", par),
                     ("Lee et al. data-consistent (0.5/yr, 7d)", dict(par, spk_lam=0.5, spk_hl=7.0)),
                     ("12/yr, 7d", dict(par, spk_hl=7.0)), ("0.5/yr, 2d", dict(par, spk_lam=0.5))):
        lo, hi = rr_threshold("avg", INF, p, pp), rr_threshold("fin", INF, p, pp)
        mid = 0.5 * (lo + hi)
        alt.append(dict(spikes=name, avg_boundary=lo, fin_boundary=hi, mid=mid,
                        rr25_fin_at_mid=100 * smile_theory("fin", mid, INF, p, pp)["rr25"],
                        rr25_avg_at_mid=100 * smile_theory("avg", mid, INF, p, pp)["rr25"],
                        rr25_fin_no_launch=100 * smile_theory("fin", 0.0, INF, p, pp)["rr25"],
                        rr25_avg_no_launch=100 * smile_theory("avg", 0.0, INF, p, pp)["rr25"]))
        print("   band:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in alt[-1].items()})
    res["band_by_spike_setting"] = alt

    # ---- E. launch probability, with and without spikes -----------------------------
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
    prob_rows = []
    for spk, pp in (("baseline spikes", par), ("no spikes", dict(par, spk_lam=0.0))):
        for q in probs:
            prob_rows.append(dict(spikes=spk, p=q,
                                  rr25_avg=rr_threshold("avg", INF, q, pp), rr25_fin=rr_threshold("fin", INF, q, pp),
                                  rule_avg=cumulant_rule("avg", INF, q, pp)["mstar"], rule_fin=cumulant_rule("fin", INF, q, pp)["mstar"],
                                  c3_fin=cumulant_rule("fin", INF, q, pp)["c3"], c3_avg=cumulant_rule("avg", INF, q, pp)["c3"]))
    res["by_launch_probability"] = prob_rows
    s = par["mag_cv"]
    a_, b_, c_ = 2.0, -3 * math.exp(s * s), math.exp(3 * s * s)
    res["p_critical_final"] = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    print(f"[{time.time() - t0:5.1f}s] critical launch probability for the final print: {res['p_critical_final']:.3f}")
    for r in prob_rows:
        print(f"   {r['spikes']:<16} p={r['p']:.2f}  avg RR25 m*={r['rr25_avg']:.3f} (rule {r['rule_avg']:.3f})  "
              f"final RR25 m*={r['rr25_fin']:.3f} (rule {r['rule_fin']:.3f})")
    # Monte Carlo test of the no-spike prediction
    nospike = dict(par, spk_lam=0.0)
    blocks0 = [build_block(SEED + 300 + i, 25_000, nospike) for i in range(16)]
    mc_ns = []
    for q, mm in ((0.8, 0.10), (0.8, 0.20), (0.9, 0.10), (0.6, 0.10)):
        st = scenario_stats(settlements(blocks0, mm, INF, q, 1, 30, par["mag_cv"]))
        mc_ns.append(dict(p=q, m=mm, mc_fin=100 * st["rr25_fin"], mc_fin_se=100 * st["rr25_fin_se"],
                          mc_avg=100 * st["rr25_avg"], mc_avg_se=100 * st["rr25_avg_se"], mc_flip=st["flip"],
                          th_fin=100 * smile_theory("fin", mm, INF, q, nospike)["rr25"],
                          th_avg=100 * smile_theory("avg", mm, INF, q, nospike)["rr25"]))
        print("   no spikes MC:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mc_ns[-1].items()})
    res["no_spike_mc_check"] = mc_ns

    # ---- F. decay through the month (no launch yet), theory vs conditional MC --------
    days = list(range(0, 28))
    decay = dict(days=days)
    for tag, pf in (("updated", lambda t: p_updated(p, t)), ("held", lambda t: p)):
        for kind in ("fin", "avg"):
            decay[f"{tag}_{kind}"] = [100 * smile_theory(kind, 0.10, INF, pf(t), par, t)["rr25"] for t in days]
            decay[f"{tag}_{kind}_rule_mstar"] = [cumulant_rule(kind, INF, pf(t), par, t)["mstar"] for t in days]
            decay[f"{tag}_{kind}_rr25_mstar"] = [rr_threshold(kind, INF, pf(t), par, t) for t in (0, 5, 10, 15, 20)]
    res["decay"] = decay
    cpath = os.path.join(OUT, "conditional_repricing.json")
    mc_cond = json.load(open(cpath)) if os.path.exists(cpath) else None
    if mc_cond:
        cmp_rows = []
        for tag in ("updated", "held"):
            for t in (0, 5, 10, 15, 20, 25):
                r = mc_cond[tag][str(t)]
                cmp_rows.append(dict(case=tag, day=t,
                                     mc_fin=100 * r["rr25_fin"], th_fin=decay[f"{tag}_fin"][t],
                                     mc_avg=100 * r["rr25_avg"], th_avg=decay[f"{tag}_avg"][t]))
                print("   decay:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cmp_rows[-1].items()})
        res["decay_vs_mc"] = cmp_rows
    n_left = np.array([N_DAYS - t for t in days], float)
    res["decay_power_law_avg_held"] = dict(
        note="rule m*(t)/m*(0) against (N/n)^(2/3) with the launch probability held",
        ratio=[decay["held_avg_rule_mstar"][i] / decay["held_avg_rule_mstar"][0] for i in (5, 10, 15, 20)],
        power_law=[float((N_DAYS / n_left[i]) ** (2 / 3)) for i in (5, 10, 15, 20)])
    print("   decay power law:", res["decay_power_law_avg_held"])

    # ---- figure -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    red, blue = "#b2182b", "#2166ac"
    xpos = lambda h: 120.0 if math.isinf(h) else h
    ax = axes[0]
    xs = np.array([xpos(hl_of(b["hl"])) for b in bound])
    cap = lambda v: np.where(np.isfinite(v), v, np.nan)
    for kind, col, lab in (("avg", blue, "monthly average"), ("fin", red, "final daily print")):
        ax.plot(xs, cap(np.array([b[f"rr25_{kind}"] for b in bound])), color=col, lw=2, label=f"{lab}: 25Δ RR = 0 (semi-analytic)")
        ax.plot(xs, cap(np.array([b[f"rule_{kind}"] for b in bound])), color=col, lw=1.3, ls="--", label=f"{lab}: cumulant rule")
        mx = [(xpos(hl_of(b["hl"])), b[f"mc_{kind}"]) for b in bound if f"mc_{kind}" in b and math.isfinite(b[f"mc_{kind}"])]
        ax.plot([a for a, _ in mx], [b for _, b in mx], ls="none", marker="o", mfc="white", mec=col, mew=1.8, ms=8,
                label=f"{lab}: Monte Carlo zero crossing")
    ya = np.array([b["rr25_avg"] for b in bound])
    yf = np.array([b["rr25_fin"] for b in bound])
    ok = np.isfinite(ya) & np.isfinite(yf)
    ax.fill_between(xs, ya, yf, where=ok & (ya < yf), color="0.82", zorder=0, label="sign-flip band")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 30, 120], ["1d", "2d", "5d", "10d", "30d", "persists"])
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("markdown half-life")
    ax.set_ylabel("mean launch markdown at which skew changes sign (log units)")
    ax.set_title("Sign boundaries: formula vs simulation", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key, col, lab in (("avg", blue, "monthly average"), ("fin", red, "final daily print")):
        pts = [(c[f"rule_{key}"], c[f"rr25_{key}"]) for c in coll if math.isfinite(c[f"rr25_{key}"])]
        ax.plot([a for a, _ in pts], [b for _, b in pts], ls="none", marker="o", ms=5, color=col, alpha=0.75, label=lab)
    lim = [0.02, 0.7]
    ax.plot(lim, lim, color="black", lw=1, label="25Δ boundary = cumulant rule")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("cumulant-rule threshold  m* = (κ₃ spike / c)^(1/3)")
    ax.set_ylabel("semi-analytic 25Δ threshold")
    ax.set_title("36 spike settings (frequency 3–24/yr, size 0.10–0.20, half-life 1–4d)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    for kind, col, lab in (("fin", red, "final daily print"), ("avg", blue, "monthly average")):
        ax.plot(days, decay[f"updated_{kind}"], color=col, lw=2, label=f"{lab}: formula, probability updated")
        ax.plot(days, decay[f"held_{kind}"], color=col, lw=1.3, ls="--", label=f"{lab}: formula, probability held")
        if mc_cond:
            for tag, fill in (("updated", col), ("held", "white")):
                td = [0, 5, 10, 15, 20, 25]
                ax.errorbar(td, [100 * mc_cond[tag][str(t)][f"rr25_{kind}"] for t in td],
                            yerr=[100 * Z_CRIT * mc_cond[tag][str(t)][f"rr25_{kind}_se"] for t in td],
                            ls="none", marker="o" if tag == "updated" else "s", mfc=fill, mec=col, ecolor=col, ms=7, capsize=3,
                            label=f"{lab}: Monte Carlo, {tag}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("valuation day (launch not yet seen)")
    ax.set_ylabel("25Δ risk reversal, vol points")
    ax.set_title("Skew decay through the month, persistent 0.10 markdown", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    fig.suptitle("Closed-form sign boundary for the two hypothetical H100 options (assumed pricing measure, illustrative parameters)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "boundary_theory_figure.png"), dpi=160)
    res["runtime_seconds"] = time.time() - t0
    with open(os.path.join(OUT, "boundary_theory.json"), "w") as f:
        json.dump(clean(res), f, indent=1)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
