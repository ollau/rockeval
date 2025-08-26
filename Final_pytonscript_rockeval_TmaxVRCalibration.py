#!/usr/bin/env python3
# Invert source-rock kinetics from RockEval (HI, Tmax), and make BurnhamDL geologic HI–vs–%VR curves.
# - Variable-rate RockEval profile (preheat + pseudo-hold + main ramp)
# - Inversion for f(E) using HI & Tmax
# - f(E) written as density and F_E_fraction (∑≈1) + BAR plot vs kcal/mol
# - BurnhamDL DAEM + Easy%RoDL → %VR for a linear geologic heating history
# - HI–vs–%VR curves use F from the inversion kinetics (F_fit)
# - HI vs Tmax plot uses HI (not normalized)
# - Summary figure includes: f(E) bar plot, HI vs Tmax (HI), Modelled vs Measured Tmax,
#   HI fraction (meas vs modelled), VR history, HI vs VR (curves), TR vs T, TR vs Time,
#   kinetics table with caption below, and a footer with the executed command line.

import argparse, sys, textwrap
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

R = 8.3145  # J/mol-K

# ---------------- Helpers & IO ----------------
def huber(r, delta=1.5):
    a = np.abs(r)
    return np.where(a <= delta, 0.5*r*r, delta*(a - 0.5*delta))

def _colmap(df):
    return {c.lower(): c for c in df.columns}

def read_any(csv_path: str):
    # auto-detect delimiter
    df = pd.read_csv(csv_path, sep=None, engine="python")
    cmap = _colmap(df)

    if {"hi","tmax","s1","s2"} <= set(cmap.keys()):
        return (df[cmap["hi"]].astype(float).values,
                df[cmap["tmax"]].astype(float).values,
                df[cmap["s1"]].astype(float).values,
                df[cmap["s2"]].astype(float).values, df)

    if {"toc","s1","s2","s3","tmax","oi","hi"} <= set(cmap.keys()):
        return (df[cmap["hi"]].astype(float).values,
                df[cmap["tmax"]].astype(float).values,
                df[cmap["s1"]].astype(float).values,
                df[cmap["s2"]].astype(float).values, df)

    raise ValueError("CSV must have either: hi,tmax,s1,s2 OR TOC,S1,S2,S3,Tmax,OI,HI")

# ---------------- Variable-rate RockEval profile ----------------
def make_temperature_profile_with_hold(
    T_start_C: float,
    T_pre_end_C: float,
    T_end_C: float,
    beta_pre_C_per_min: float,
    beta_main_C_per_min: float,
    hold_minutes: float = 3.0,
    dT: float = 1.0,
    hold_delta_T: float = 0.5,
):
    # Segment 1: preheat
    T1 = np.arange(T_start_C, T_pre_end_C, dT)
    # Segment 2: pseudo-hold via tiny ramp lasting 'hold_minutes'
    beta_hold = max(hold_delta_T / max(hold_minutes, 1e-9), 1e-6)
    T2_start = T_pre_end_C
    T2_end   = T_pre_end_C + hold_delta_T
    n2 = max(int(np.ceil((T2_end - T2_start) / max(dT, 1e-6))) + 1, 2)
    T2 = np.linspace(T2_start, T2_end, n2)
    # Segment 3: main ramp
    T3 = np.arange(T2[-1], T_end_C + dT, dT)

    # Concatenate (avoid duplicates at joins)
    if T1.size == 0:
        T_C = np.concatenate([T2, T3[1:]])
    else:
        T_C = np.concatenate([T1, T2[1:], T3[1:]])

    beta_profile = np.empty_like(T_C)
    beta_profile[:] = beta_main_C_per_min
    beta_profile[T_C < T_pre_end_C] = beta_pre_C_per_min
    beta_profile[(T_C >= T_pre_end_C) & (T_C <= T2_end)] = beta_hold
    return T_C, beta_profile

def build_kernel_variable(
    T_C: np.ndarray,
    E_Jmol: np.ndarray,
    A_s: float,
    beta_profile_C_per_min: np.ndarray,
) -> np.ndarray:
    """
    Vectorized kernel for arbitrary T(t) with pointwise heating rate β(T).
    Returns K[T,E] = dα/dT normalized over T for each E.
    """
    T_K = T_C + 273.15
    beta_Ks = np.maximum(beta_profile_C_per_min, 1e-12) / 60.0  # K/s

    invRT = (1.0 / (R * T_K))[:, None]               # [nT,1]
    kTE = A_s * np.exp(-invRT * E_Jmol[None, :])     # [nT,nE]
    integrand = kTE / beta_Ks[:, None]               # per K

    # cumulative ∫(k/β) dT in T for each E (trapezoid)
    dT = np.diff(T_C, prepend=T_C[0])
    G = np.cumsum(0.5 * (integrand[1:, :] + integrand[:-1, :]) * dT[1:, None], axis=0)
    G = np.vstack([np.zeros((1, integrand.shape[1])), G])

    rate_T = integrand * np.exp(-G)                  # dα/dT
    area = np.trapezoid(rate_T, T_C, axis=0)
    area = np.where(area > 0, area, 1.0)
    K = rate_T / area[None, :]
    return K

# ---------------- Inversion outputs ----------------
@dataclass
class FitResult:
    E_kJmol: np.ndarray
    f: np.ndarray
    HI0: float
    m: np.ndarray
    T_pred_C: np.ndarray
    R_pred: np.ndarray
    rmse_Tmax: float
    rmse_R: float

# ---------------- Jarvie empirical VR for overlay ----------------
def jarvie_vr(tmax_C: np.ndarray) -> np.ndarray:
    """
    Jarvie Tmax → %VR:
        %VR = 0.018 * Tmax - 7.16     (clipped 0.2–5.0)
    """
    vr = 0.018 * np.asarray(tmax_C, dtype=float) - 7.16
    return np.clip(vr, 0.2, 5.0)

# ---------------- Inversion (HI & Tmax) ----------------
def invert_hi_tmax(
    HI_obs, Tmax_obs_C,
    *, beta_C_per_min=25.0, A_s=1e14,
    Emin_kJ=130.0, Emax_kJ=320.0, nE=41,
    Tgrid_C=(200.0, 640.0, 901),
    lambda_smooth=0.2, w_tmax=1.0, w_hi=1.0,
    fit_HI0=True, HI0_fixed=None,
    preheat_rate=50.0, preheat_start=20.0, preheat_end=300.0, hold_minutes=3.0,
    tmax_cut=330.0,
) -> FitResult:

    n = len(HI_obs)

    # Energy grid
    E_kJ = np.linspace(Emin_kJ, Emax_kJ, nE)
    E_J  = E_kJ * 1e3

    # Variable-rate RockEval program
    T_start = preheat_start
    T_pre_end = preheat_end
    T_end = Tgrid_C[1]
    dT_default = max((T_end - T_start) / max(Tgrid_C[2], 400), 1.0)
    T_C, beta_prof = make_temperature_profile_with_hold(
        T_start_C=T_start, T_pre_end_C=T_pre_end, T_end_C=T_end,
        beta_pre_C_per_min=preheat_rate, beta_main_C_per_min=beta_C_per_min,
        hold_minutes=hold_minutes, dT=dT_default
    )
    K = build_kernel_variable(T_C, E_J, A_s, beta_prof)

    # q(E): selector; lower E depletes faster
    Escale = (Emax_kJ - Emin_kJ) / 6.0
    q = np.exp(-(E_kJ - Emin_kJ) / max(Escale, 1e-6))

    # Initial guesses
    HI0_guess = float(np.nanmax(HI_obs)) if HI0_fixed is None else float(HI0_fixed)
    mu = 0.5*(Emin_kJ+Emax_kJ)
    sig = 0.15*(Emax_kJ-Emin_kJ)
    f0 = np.exp(-0.5*((E_kJ-mu)/sig)**2)
    f0 = f0 / (np.trapezoid(f0, E_kJ) + 1e-12)

    R_obs0 = np.clip(HI_obs / max(HI0_guess, 1e-9), 1e-6, 1.0)
    m0 = -np.log(R_obs0) / (np.average(q, weights=f0) + 1e-12)
    m0 = np.clip(m0, 0.0, 100.0)

    # softplus helpers
    sp = lambda z: np.log1p(np.exp(z))
    inv_sp = lambda y: np.log(np.expm1(np.clip(y, 1e-12, None)))

    # pack/unpack
    u0 = inv_sp(f0 / (np.max(f0)+1e-12))
    v0 = inv_sp(m0 + 1e-6)
    x0 = np.concatenate([u0, v0, [inv_sp(HI0_guess)]]) if fit_HI0 else np.concatenate([u0, v0])

    # E-integration weights
    wE = np.zeros_like(E_kJ)
    wE[1:-1] = (E_kJ[2:] - E_kJ[:-2]) / 2.0
    wE[0]  = (E_kJ[1] - E_kJ[0]) / 2.0
    wE[-1] = (E_kJ[-1] - E_kJ[-2]) / 2.0

    # residual scales
    T_scale = max(np.std(Tmax_obs_C), 1.0)
    R_obs_tmp = np.clip(HI_obs / max(HI0_guess, 1e-9), 1e-6, 1.5)
    R_scale = max(np.std(R_obs_tmp), 1e-3)

    # Tmax cutoff mask
    tmask = (T_C >= tmax_cut)
    use_cut = np.any(tmask)

    def argmax_with_cut(S2n_row):
        if use_cut:
            idx = np.argmax(S2n_row[tmask])
            return T_C[tmask][idx]
        else:
            return T_C[np.argmax(S2n_row)]

    def objective(x):
        # unpack
        if fit_HI0:
            u = x[:nE]; v = x[nE:nE+n]; h = x[-1]
        else:
            u = x[:nE]; v = x[nE:nE+n]; h = None

        f_raw = sp(u)
        f = f_raw / (np.trapezoid(f_raw, E_kJ) + 1e-12)
        m = sp(v)
        HI0 = sp(h) if fit_HI0 else HI0_fixed

        Q = np.exp(-np.outer(m, q))      # [n, nE]
        R_pred = (Q @ (f * wE))
        R_obs = np.clip(HI_obs / np.maximum(HI0, 1e-9), 1e-6, 1.5)

        S2 = (Q * f) @ K.T
        area_T = np.trapezoid(S2, T_C, axis=1) + 1e-12
        S2n = (S2.T / area_T).T
        Tmax_pred = np.array([argmax_with_cut(S2n[i, :]) for i in range(S2n.shape[0])])

        rT = (Tmax_pred - Tmax_obs_C) / T_scale
        rR = (R_pred - R_obs) / R_scale
        J_T = np.mean(huber(rT, 1.5))
        J_R = np.mean(huber(rR, 1.5))

        # smoothness on f
        d2f = np.diff(f, n=2)
        reg = lambda_smooth * np.dot(d2f, d2f)

        return w_tmax * J_T + w_hi * J_R + reg

    res = minimize(objective, x0, method="L-BFGS-B",
                   options={"maxiter": 600, "ftol": 1e-9})

    # Unpack & diagnostics
    if fit_HI0:
        u = res.x[:nE]; v = res.x[nE:nE+n]; h = res.x[-1]
    else:
        u = res.x[:nE]; v = res.x[nE:nE+n]; h = None

    f_raw = np.log1p(np.exp(u))
    f = f_raw / (np.trapezoid(f_raw, E_kJ) + 1e-12)
    m = np.log1p(np.exp(v))
    HI0 = np.log1p(np.exp(h)) if fit_HI0 else HI0_fixed

    Q = np.exp(-np.outer(m, q))
    R_pred = (Q @ (f * wE))

    S2 = (Q * f) @ K.T
    area_T = np.trapezoid(S2, T_C, axis=1) + 1e-12
    S2n = (S2.T / area_T).T
    if use_cut:
        Tmax_pred = np.array([T_C[tmask][np.argmax(S2n[i, tmask])] for i in range(S2n.shape[0])])
    else:
        Tmax_pred = T_C[np.argmax(S2n, axis=1)]

    rmse_Tmax = float(np.sqrt(np.mean((Tmax_pred - Tmax_obs_C)**2)))
    R_obs = np.clip(HI_obs / max(HI0, 1e-9), 1e-6, 1.5)
    rmse_R = float(np.sqrt(np.mean((R_pred - R_obs)**2)))

    return FitResult(E_kJ, f, HI0, m, Tmax_pred, R_pred, rmse_Tmax, rmse_R)

# ---------------- BurnhamDL DAEM + Easy%RoDL for %VR ----------------
BURNHAMDL_E_KCAL = np.array([
    38,40,42,44,46,48,50,52,54,56,
    58,60,62,64,66,68,70,72,74,76
], dtype=float)
BURNHAMDL_W_PCT = np.array([
    2.11,3.16,4.21,5.26,4.21,4.21,3.16,4.21,7.37,10.53,
    9.47,7.37,6.32,5.26,5.26,4.21,4.21,3.16,3.16,3.16
], dtype=float)
BURNHAMDL_A_S = 2.00152e14
BURNHAMDL_W_FRAC = BURNHAMDL_W_PCT / np.sum(BURNHAMDL_W_PCT)

def vro_burnhamdl_from_history(
    T_hist_C, dt_hist_s,
    *, E_kcal=BURNHAMDL_E_KCAL, w_frac=BURNHAMDL_W_FRAC, A_s=BURNHAMDL_A_S,
    return_F: bool = False
):
    """
    Compute %VR along T–t using BurnhamDL DAEM and Easy%RoDL mapping:
        %Ro = exp(-1.5 + 3.7 * F)
    If return_F=True, return (Ro, F).
    """
    R_cal = 1.987  # cal/mol-K
    T_K = T_hist_C + 273.15
    invRT = (1.0 / (R_cal * T_K))[:, None]

    # Arrhenius k(T) for each Ea bin (E kcal/mol -> cal/mol)
    kTE = A_s * np.exp(-(E_kcal[None, :] * 1000.0) * invRT)  # [nT, nE]

    dt = dt_hist_s.copy()
    if dt.size:
        dt[0] = 0.0

    # cumulative ∫k dt (trapezoid in time)
    G = np.zeros_like(kTE)
    if T_hist_C.size >= 2:
        kd = 0.5 * (kTE[1:] + kTE[:-1]) * dt[1:, None]
        G[1:] = np.cumsum(kd, axis=0)

    x = 1.0 - np.exp(-G)     # conversion per bin
    F = x @ w_frac           # reacted fraction
    Ro = np.exp(-1.5 + 3.7 * F)
    Ro = np.clip(Ro, 0.2, 7.0)
    return (Ro, F) if return_F else Ro

def build_geologic_history_linear(T0_C, Tend_C, rate_C_per_Ma, dt_ka=10.0):
    """Linear geologic heating from T0 to Tend at 'rate' °C/Ma; returns T_hist, dt_hist_s."""
    if rate_C_per_Ma <= 0:
        raise ValueError("rate_C_per_Ma must be > 0")
    total_dT = max(Tend_C - T0_C, 0.0)
    total_Ma = total_dT / rate_C_per_Ma
    if total_Ma == 0:
        return np.array([T0_C]), np.array([0.0])

    n_steps = max(int(np.ceil(total_Ma * 1000.0 / dt_ka)), 2)
    t_ka = np.linspace(0.0, total_Ma * 1000.0, n_steps)
    T_hist = T0_C + (rate_C_per_Ma / 1000.0) * t_ka
    ka_to_s = 1000.0 * 365.25 * 24 * 3600
    dt_s = np.diff(t_ka, prepend=t_ka[0]) * ka_to_s
    return T_hist, dt_s

# ---------------- Combined summary figure ----------------
def make_all_plots_summary(E_kcal, F_E_fraction, Tmax, HI, fit,
                           T_hist, VR_hist, F_fit, TR_percent, t_Ma, args):

    # Vector fonts in PDF
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42

    fig, axs = plt.subplots(3, 3, figsize=(15, 16), constrained_layout=False)

    # (1) f(E) BAR plot
    bar_w = 0.9 * np.median(np.diff(E_kcal)) if len(E_kcal) > 1 else 1.0
    axs[0,0].bar(E_kcal, F_E_fraction, width=bar_w, align="center", edgecolor="black")
    axs[0,0].set_xlabel("E (kcal/mol)")
    axs[0,0].set_ylabel("F_E_fraction")
    axs[0,0].set_title("f(E) distribution")

    # (2) HI vs measured Tmax (Y = HI, not normalized)
    axs[0,1].scatter(Tmax, HI, c="k")
    axs[0,1].set_xlabel("Measured Tmax (°C)")
    axs[0,1].set_ylabel("HI (mg HC/g TOC)")
    axs[0,1].set_title("HI vs measured Tmax")

    # (3) Modelled vs Measured Tmax
    axs[0,2].scatter(Tmax, fit.T_pred_C, c="blue")
    lims = [min(Tmax.min(), fit.T_pred_C.min()) - 5,
            max(Tmax.max(), fit.T_pred_C.max()) + 5]
    axs[0,2].plot(lims, lims, "r--")
    axs[0,2].axvline(args.tmax_cut, ls="--", lw=1, color="grey")
    axs[0,2].set_xlabel("Measured Tmax (°C)")
    axs[0,2].set_ylabel("Modelled Tmax (°C)")
    axs[0,2].set_title("Modelled vs Measured Tmax")

    # (4) HI fraction vs Tmax: measured vs modelled (still included)
    frac_meas = HI / fit.HI0
    frac_pred = fit.R_pred
    axs[1,0].scatter(Tmax, frac_meas, c="k", label="Measured")
    axs[1,0].scatter(fit.T_pred_C, frac_pred, c="r", marker="x", label="Modelled")
    for xm, ym, xp, yp in zip(Tmax, frac_meas, fit.T_pred_C, frac_pred):
        axs[1,0].plot([xm, xp], [ym, yp], "grey", lw=0.7, alpha=0.6)
    axs[1,0].set_xlabel("Tmax (°C)")
    axs[1,0].set_ylabel("HI/HI0")
    axs[1,0].set_title("HI fraction: Measured vs Modelled")
    axs[1,0].legend(fontsize=8)

    # (5) Geologic VR history
    axs[1,1].plot(T_hist, VR_hist)
    axs[1,1].set_xlabel("Temperature (°C)")
    axs[1,1].set_ylabel("%VR (BurnhamDL)")
    axs[1,1].set_title("Geologic VR history")
    axs[1,1].text(
        0.5, -0.25, f"{args.geo_rate} °C/Ma linear heating rate",
        transform=axs[1,1].transAxes, ha="center", va="top", fontsize=8
    )

    # (6) HI vs %VR (curves + measured) — F_fit drives HI decay
    for HI0v in [200, 300, 400, 500, 600, 700]:
        if args.hi_decay == "linear":
            HI_curve = HI0v * (1.0 - F_fit)
        else:
            # normalized exponential, HI(F=1)=0
            k = float(args.hi_k)
            denom = 1.0 - np.exp(-k)
            denom = max(denom, 1e-12)
            HI_curve = HI0v * (np.exp(-k * F_fit) - np.exp(-k)) / denom
        axs[1,2].plot(VR_hist, HI_curve, label=f"HI0={int(HI0v)}")
    VR_jarvie = jarvie_vr(Tmax)
    axs[1,2].scatter(VR_jarvie, HI, c="k", s=30, zorder=5, label="Measured")
    axs[1,2].set_xlim(0.2, 7.0)
    axs[1,2].set_xlabel("%VR (geologic)")
    axs[1,2].set_ylabel("HI (mg HC/g TOC)")
    axs[1,2].set_title("HI vs %VR (curves)")
    axs[1,2].legend(fontsize=7)
    axs[1,2].text(
        0.5, -0.25, f"{args.geo_rate} °C/Ma linear heating rate",
        transform=axs[1,2].transAxes, ha="center", va="top", fontsize=8
    )

    # (7) TR vs T
    axs[2,0].plot(T_hist, TR_percent)
    axs[2,0].set_xlabel("Temperature (°C)")
    axs[2,0].set_ylabel("TR (%)")
    axs[2,0].set_ylim(0, 100)
    axs[2,0].set_title("Transformation Ratio vs Temperature")
    axs[2,0].text(
        0.5, -0.25, f"{args.geo_rate} °C/Ma linear heating rate",
        transform=axs[2,0].transAxes, ha="center", va="top", fontsize=8
    )

    # (8) TR vs Time
    axs[2,1].plot(t_Ma, TR_percent)
    axs[2,1].set_xlabel("Time (Ma)")
    axs[2,1].set_ylabel("TR (%)")
    axs[2,1].set_ylim(0, 100)
    axs[2,1].set_title("Transformation Ratio vs Time")
    axs[2,1].text(
        0.5, -0.25, f"{args.geo_rate} °C/Ma linear heating rate",
        transform=axs[2,1].transAxes, ha="center", va="top", fontsize=8
    )

    # (9) Kinetics table + caption BELOW the table
    axs[2,2].axis("off")
    cell_text = [[f"{E:.1f}", f"{fval:.4f}"] for E, fval in zip(E_kcal, F_E_fraction)]
    tbl = axs[2,2].table(
        cellText=cell_text,
        colLabels=["E_kcal/mol", "F_E_fraction"],
        loc="center", cellLoc="center", colLoc="center"
    )
    # dynamic font size
    n_rows = len(cell_text)
    fsize = 8 if n_rows <= 15 else (6 if n_rows <= 30 else 5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fsize)
    tbl.scale(1.0, 1.0)

    # ensure canvas so we can compute extents
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Find the lowest cell (smallest y0 in axes coords)
    min_y = 1.0
    for ((r, c), cell) in tbl.get_celld().items():
        bbox_disp = cell.get_window_extent(renderer=renderer)
        bbox_axes = bbox_disp.transformed(axs[2,2].transAxes.inverted())
        min_y = min(min_y, bbox_axes.y0)
    y_caption = min_y - 0.05  # small gap under table
    axs[2,2].text(
        0.5, y_caption,
        "Inversion kinetics: Ea (kcal/mol) and fraction (0–1). Constant A: 1E14/s.",
        ha="center", va="top", fontsize=8, transform=axs[2,2].transAxes
    )

    # Footer with command line, wrapped
    run_line = " ".join(sys.argv)
    def wrap_for_figure(text, fig, fontsize=7, char_aspect=0.55):
        dpi = fig.dpi
        fig_w_in = fig.get_figwidth()
        char_px = char_aspect * fontsize * (dpi / 72.0)
        line_px = fig_w_in * dpi
        width_chars = max(30, int(line_px / char_px))
        return "\n".join(textwrap.wrap(text, width=width_chars))
    footer_text = wrap_for_figure(f"Run: {run_line}", fig, fontsize=7)
    plt.subplots_adjust(bottom=0.18)
    fig.text(0.01, 0.01, footer_text, ha="left", va="bottom", fontsize=7)

    # Save (PNG + PDF)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("all_plots_summary.png", dpi=300)
    with PdfPages("all_plots_summary.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print("Wrote combined plots: all_plots_summary.png and all_plots_summary.pdf")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input RockEval CSV")
    # Inversion controls
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--A", type=float, default=1e14)
    ap.add_argument("--emin", type=float, default=130.0)
    ap.add_argument("--emax", type=float, default=320.0)
    ap.add_argument("--nE", type=int, default=41)
    ap.add_argument("--tmin", type=float, default=200.0)
    ap.add_argument("--tmax", type=float, default=650.0)
    ap.add_argument("--nT", type=int, default=901)
    ap.add_argument("--lambda_smooth", type=float, default=0.2)
    ap.add_argument("--w_tmax", type=float, default=1.0)
    ap.add_argument("--w_hi", type=float, default=1.0)
    ap.add_argument("--tmax_cut", type=float, default=330.0, help="Lower T cutoff when finding Tmax (°C)")
    ap.add_argument("--fix_HI0", type=str, default="None",
                    help="Fix HI0 (mg HC/g TOC). Use 'None' to let inversion estimate HI0.")
    ap.add_argument("--seed", type=int, default=42)

    # RockEval preheat/hold
    ap.add_argument("--preheat_rate", type=float, default=50.0)
    ap.add_argument("--preheat_start", type=float, default=20.0)
    ap.add_argument("--preheat_end", type=float, default=300.0)
    ap.add_argument("--hold_minutes", type=float, default=3.0)

    # Geologic history (for %VR axis)
    ap.add_argument("--geo_T0", type=float, default=20.0)
    ap.add_argument("--geo_Tend", type=float, default=300.0)
    ap.add_argument("--geo_rate", type=float, default=2.0, help="°C/Ma")
    ap.add_argument("--geo_dt_ka", type=float, default=10.0)

    # HI decay options for HI vs %VR curves
    ap.add_argument("--hi_decay", choices=["linear","exp"], default="exp",
                    help="Decay law for HI(F_fit): 'linear' = HI0*(1-F); 'exp' = normalized exp so HI(F=1)=0")
    ap.add_argument("--hi_k", type=float, default=4.5,
                    help="Exponent k for exponential HI decay (used when --hi_decay exp)")

    args = ap.parse_args()
    fix_HI0_val = None if str(args.fix_HI0).strip().lower() == "none" else float(args.fix_HI0)

    # Read data
    HI, Tmax, S1, S2, df = read_any(args.csv)

    # ---- Inversion ----
    fit = invert_hi_tmax(
        HI_obs=HI, Tmax_obs_C=Tmax, beta_C_per_min=args.beta,
        A_s=args.A, Emin_kJ=args.emin, Emax_kJ=args.emax, nE=args.nE,
        Tgrid_C=(args.tmin, args.tmax, args.nT),
        lambda_smooth=args.lambda_smooth,
        w_tmax=args.w_tmax, w_hi=args.w_hi,
        fit_HI0=(fix_HI0_val is None), HI0_fixed=fix_HI0_val,
        preheat_rate=args.preheat_rate, preheat_start=args.preheat_start,
        preheat_end=args.preheat_end, hold_minutes=args.hold_minutes,
        tmax_cut=args.tmax_cut,
    )

    print("\n=== Inversion summary ===")
    print(f"Samples: {len(HI)}")
    print(f"Preheat: {args.preheat_start}→{args.preheat_end} °C @ {args.preheat_rate} °C/min; hold {args.hold_minutes} min")
    print(f"Main ramp: 300→{args.tmax:.0f} °C @ {args.beta:.1f} °C/min | Tmax cut: {args.tmax_cut:.1f} °C")
    print(f"A_s={args.A:.2e} s^-1 | HI0={fit.HI0:.1f} mgHC/gTOC")
    print(f"RMSE Tmax: {fit.rmse_Tmax:.2f} °C | RMSE HI/HI0: {fit.rmse_R:.4f}")

    # Per-sample outputs
    out = df.copy()
    out["HI_pred"] = fit.R_pred * fit.HI0
    out["Tmax_pred_C"] = fit.T_pred_C
    out["maturity_scalar_m"] = fit.m
    out.to_csv("hi_tmax_fit_by_sample.csv", index=False)
    print("Wrote: hi_tmax_fit_by_sample.csv")

    # f(E) density and fractions
    E_kJ = fit.E_kJmol.copy()
    f = fit.f.copy()
    # numerical E-weights for fractions
    wE = np.zeros_like(E_kJ)
    wE[1:-1] = (E_kJ[2:] - E_kJ[:-2]) / 2.0
    wE[0]  = (E_kJ[1] - E_kJ[0]) / 2.0
    wE[-1] = (E_kJ[-1] - E_kJ[-2]) / 2.0
    F_E_fraction = f * wE
    E_kcal = E_kJ / 4.184

    f_df = pd.DataFrame({
        "E_kJmol": E_kJ,
        "E_kcalmol": E_kcal,
        "f_E_density": f,
        "F_E_fraction": F_E_fraction
    })
    f_df.to_csv("fE_distribution.csv", index=False)
    print(f"Check ∫f(E)dE via trapezoid: {np.trapezoid(f, E_kJ):.6f}")
    print(f"Check sum(F_E_fraction): {np.sum(F_E_fraction):.6f}")
    print("Wrote: fE_distribution.csv")

    # ---- BurnhamDL geologic %VR (x-axis) + F_fit from inversion (for HI curves) ----
    T_hist, dt_hist_s = build_geologic_history_linear(
        args.geo_T0, args.geo_Tend, args.geo_rate, args.geo_dt_ka
    )
    VR_hist, F_hist_bdl = vro_burnhamdl_from_history(T_hist, dt_hist_s, return_F=True)

    # Compute F_fit (reacted fraction) from *inversion* kinetics along same geologic history
    E_kJ_fit = fit.E_kJmol.copy()
    f_density = fit.f.copy()
    wE_fit = np.zeros_like(E_kJ_fit)
    wE_fit[1:-1] = (E_kJ_fit[2:] - E_kJ_fit[:-2]) / 2.0
    wE_fit[0]  = (E_kJ_fit[1] - E_kJ_fit[0]) / 2.0
    wE_fit[-1] = (E_kJ_fit[-1] - E_kJ_fit[-2]) / 2.0
    w_frac_fit = f_density * wE_fit
    ws = w_frac_fit.sum()
    if ws <= 0: ws = 1.0
    w_frac_fit = w_frac_fit / ws

    E_J_fit = E_kJ_fit * 1e3
    T_K_hist = T_hist + 273.15
    invRT = (1.0 / (R * T_K_hist))[:, None]
    kTE_fit = args.A * np.exp(-invRT * E_J_fit[None, :])

    dt_s = dt_hist_s.copy()
    if dt_s.size: dt_s[0] = 0.0
    G_fit = np.zeros_like(kTE_fit)
    if T_hist.size >= 2:
        kd = 0.5 * (kTE_fit[1:, :] + kTE_fit[:-1, :]) * dt_s[1:, None]
        G_fit[1:, :] = np.cumsum(kd, axis=0)

    x_fit = 1.0 - np.exp(-G_fit)
    F_fit = np.clip(x_fit @ w_frac_fit, 0.0, 1.0)

    # TR and time axis
    TR_percent = 100.0 * F_fit
    sec_per_Ma = 1_000_000.0 * 365.25 * 24.0 * 3600.0
    t_Ma = np.cumsum(dt_s) / sec_per_Ma

    # Save geologic history CSV (include both F’s)
    out_geo = pd.DataFrame({
        "T_C": T_hist,
        "VR_BurnhamDL": VR_hist,
        "F_reacted_inversion": F_fit,
        "F_reacted_BurnhamDL": F_hist_bdl,
        "TR_percent_inversion": TR_percent,
        "Time_Ma": t_Ma
    })
    out_geo.to_csv("VR_BurnhamDL_geologic_history.csv", index=False)
    print("Wrote: VR_BurnhamDL_geologic_history.csv")

    # --- Combined summary figure with all key plots ---
    make_all_plots_summary(
        E_kcal=E_kcal, F_E_fraction=F_E_fraction,
        Tmax=Tmax, HI=HI, fit=fit,
        T_hist=T_hist, VR_hist=VR_hist, F_fit=F_fit, TR_percent=TR_percent, t_Ma=t_Ma,
        args=args
    )

if __name__ == "__main__":
    main()
  