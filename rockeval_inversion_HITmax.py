#!/usr/bin/env python3
# Invert source-rock kinetics from routine RockEval (HI, Tmax) only.
# Accepts either headers: hi,tmax,s1,s2  OR  TOC,S1,S2,S3,Tmax,OI,HI
# Outputs: fE_distribution.csv  and  hi_tmax_fit_by_sample.csv


import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize

try:
    trapz = np.trapezoid  # new name
except AttributeError:
    trapz = np.trapz      # fallback for older numpy


R = 8.3145  # J/mol-K

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

def _colmap(df):
    # case-insensitive map
    return {c.lower(): c for c in df.columns}

def read_any(csv_path: str):
    """Read CSV. Support two header styles."""
    df = pd.read_csv(csv_path)
    cmap = _colmap(df)

    # Primary (your) style
    if {"toc","s1","s2","s3","tmax","oi","hi"} <= set(cmap.keys()):
        HI = df[cmap["hi"]].astype(float).values
        Tmax = df[cmap["tmax"]].astype(float).values
        S1 = df[cmap["s1"]].astype(float).values
        S2 = df[cmap["s2"]].astype(float).values
        return HI, Tmax, S1, S2, df

    # Legacy style
    if {"hi","tmax","s1","s2"} <= set(cmap.keys()):
        HI = df[cmap["hi"]].astype(float).values
        Tmax = df[cmap["tmax"]].astype(float).values
        S1 = df[cmap["s1"]].astype(float).values
        S2 = df[cmap["s2"]].astype(float).values
        return HI, Tmax, S1, S2, df

    raise ValueError(
        "CSV must have either: hi,tmax,s1,s2  OR  TOC,S1,S2,S3,Tmax,OI,HI"
    )

def build_kernel(T_C: np.ndarray, E_Jmol: np.ndarray, A_s: float, beta_C_per_min: float) -> np.ndarray:
    """Kernel K[T, E]: rate per °C for a unit-mass component at energy E, normalized to unit area over T."""
    T_K = T_C + 273.15
    beta_Ks = beta_C_per_min / 60.0  # K/s
    K = np.zeros((len(T_K), len(E_Jmol)))
    for j, E in enumerate(E_Jmol):
        kT = A_s * np.exp(-E / (R * T_K))  # s^-1
        integrand = kT / beta_Ks           # per K
        G = np.zeros_like(T_K)
        G[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(T_K))
        rate_T = (kT / beta_Ks) * np.exp(-G)    # per °C
        area = np.trapz(rate_T, T_C)
        K[:, j] = rate_T / (area if area > 0 else 1.0)
    return K

def smooth_matrix(n: int):
    L = np.zeros((n-2, n))
    for i in range(n-2):
        L[i, i]   = 1.0
        L[i, i+1] = -2.0
        L[i, i+2] = 1.0
    return L

def invert_hi_tmax(
    HI_obs: np.ndarray,
    Tmax_obs_C: np.ndarray,
    *,
    beta_C_per_min: float = 25.0,
    A_s: float = 1e14,
    Emin_kJ: float = 120.0,
    Emax_kJ: float = 320.0,
    nE: int = 41,
    Tgrid_C: Tuple[float,float,int] = (200.0, 650.0, 901),
    lambda_smooth: float = 0.2,
    w_tmax: float = 1.0,
    w_hi: float = 1.0,
    fit_HI0: bool = True,
    HI0_fixed: float = None,
    seed: int = 42
) -> FitResult:
    rng = np.random.default_rng(seed)
    n = len(HI_obs)

    # Energy & temperature grids
    E_kJ = np.linspace(Emin_kJ, Emax_kJ, nE)
    E_J = E_kJ * 1e3
    Tlo, Thi, nT = Tgrid_C
    T_C = np.linspace(Tlo, Thi, nT)
    K = build_kernel(T_C, E_J, A_s, beta_C_per_min)  # [nT, nE]

    # E-selector q(E) (low-E depletes faster)
    Escale = (Emax_kJ - Emin_kJ) / 6.0
    q = np.exp(-(E_kJ - Emin_kJ) / max(Escale, 1e-6))

    # Initial guesses
    HI0_guess = float(np.nanmax(HI_obs)) if HI0_fixed is None else float(HI0_fixed)

    mu = 0.5*(Emin_kJ+Emax_kJ)
    sig = 0.15*(Emax_kJ-Emin_kJ)
    f0 = np.exp(-0.5*((E_kJ-mu)/sig)**2)
    f0 = f0 / (np.trapz(f0, E_kJ) + 1e-12)

    R_obs0 = np.clip(HI_obs / max(HI0_guess, 1e-9), 1e-6, 1.0)
    m0 = -np.log(R_obs0) / (np.average(q, weights=f0) + 1e-12)
    m0 = np.clip(m0, 0.0, 100.0)

    # Softplus helpers
    def sp(z): return np.log1p(np.exp(z))
    def inv_sp(y): return np.log(np.expm1(np.clip(y, 1e-12, None)))

    # Pack/unpack variables: f via softplus, m via softplus, optional HI0 via softplus
    def pack(u, v, h=None):
        return np.concatenate([u, v] + (([np.array([h])]) if h is not None else []))
    def unpack(x):
        u = x[:nE]
        v = x[nE:nE+n]
        if fit_HI0:
            h = x[-1]; return u, v, h
        return u, v, None

    u0 = inv_sp(f0 / (np.max(f0)+1e-12))
    v0 = inv_sp(m0 + 1e-6)
    x0 = pack(u0, v0, inv_sp(HI0_guess)) if fit_HI0 else pack(u0, v0, None)

    L = smooth_matrix(nE)

    # Precompute trapezoid weights over E
    wE = np.zeros_like(E_kJ)
    wE[1:-1] = (E_kJ[2:] - E_kJ[:-2]) / 2.0
    wE[0] = (E_kJ[1] - E_kJ[0]) / 2.0
    wE[-1] = (E_kJ[-1] - E_kJ[-2]) / 2.0

    def objective(x):
        u, v, h = unpack(x)
        f_raw = sp(u) + 1e-12
        f = f_raw / (np.trapz(f_raw, E_kJ) + 1e-12)
        m = sp(v)
        HI0 = sp(h) if fit_HI0 else HI0_fixed

        # Remaining fractions
        Q = np.exp(-np.outer(m, q))         # [n, nE]
        R_pred = (Q @ (f * wE))             # [n]
        R_obs = np.clip(HI_obs / np.maximum(HI0, 1e-9), 1e-6, 1.5)

        # Tmax prediction from ramp
        S2 = (Q * f) @ K.T                  # [n, nT]
        area_T = np.trapz(S2, T_C, axis=1) + 1e-12
        S2n = (S2.T / area_T).T
        Tmax_pred = T_C[np.argmax(S2n, axis=1)]

        rT = Tmax_pred - Tmax_obs_C
        rR = R_pred - R_obs

        smooth = L @ f
        reg = lambda_smooth * np.sum(smooth**2)

        return w_tmax * np.mean(rT**2) + w_hi * np.mean(rR**2) + reg

    res = minimize(objective, x0, method="L-BFGS-B",
                   options={"maxiter": 600, "ftol": 1e-9})

    # Unpack solution & diagnostics
    u, v, h = unpack(res.x)
    f_raw = np.log1p(np.exp(u)) + 1e-12
    f = f_raw / (np.trapz(f_raw, E_kJ) + 1e-12)
    m = np.log1p(np.exp(v))
    HI0 = np.log1p(np.exp(h)) if fit_HI0 else HI0_fixed

    Q = np.exp(-np.outer(m, q))
    R_pred = (Q @ (f * wE))
    T_C = np.linspace(*Tgrid_C)
    K = build_kernel(T_C, E_J, A_s, beta_C_per_min)
    S2 = (Q * f) @ K.T  
    area_T = np.trapz(S2, T_C, axis=1) + 1e-12
    S2n = (S2.T / area_T).T
    T_pred = T_C[np.argmax(S2n, axis=1)]

    rmse_Tmax = float(np.sqrt(np.mean((T_pred - Tmax_obs_C)**2)))
    R_obs = np.clip(HI_obs / max(HI0,1e-9), 1e-6, 1.5)
    rmse_R = float(np.sqrt(np.mean((R_pred - R_obs)**2)))

    return FitResult(E_kJ, f, HI0, m, T_pred, R_pred, rmse_Tmax, rmse_R)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
        help="CSV with either: hi,tmax,s1,s2  OR  TOC,S1,S2,S3,Tmax,OI,HI")
    ap.add_argument("--beta", type=float, default=25.0, help="Heating rate (°C/min). Default 25")
    ap.add_argument("--A", type=float, default=1e14, help="Pre-exponential A (s^-1). Default 1e14")
    ap.add_argument("--emin", type=float, default=120.0, help="Min activation energy (kJ/mol)")
    ap.add_argument("--emax", type=float, default=320.0, help="Max activation energy (kJ/mol)")
    ap.add_argument("--nE", type=int, default=41, help="Number of energy bins")
    ap.add_argument("--tmin", type=float, default=200.0, help="Min ramp temperature (°C)")
    ap.add_argument("--tmax", type=float, default=650.0, help="Max ramp temperature (°C)")
    ap.add_argument("--nT", type=int, default=901, help="Number of T grid points")
    ap.add_argument("--lambda_smooth", type=float, default=0.2, help="Smoothness strength for f(E)")
    ap.add_argument("--w_tmax", type=float, default=1.0, help="Weight on Tmax misfit")
    ap.add_argument("--w_hi", type=float, default=1.0, help="Weight on HI fraction misfit")
    # Default HI0 fixed to 600
    ap.add_argument("--fix_HI0", type=float, default=600.0,
        help="Fix HI0 instead of fitting it (default 600 mg HC/g TOC)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    HI, Tmax, S1, S2, df = read_any(args.csv)

    fit = invert_hi_tmax(
        HI_obs=HI,
        Tmax_obs_C=Tmax,
        beta_C_per_min=args.beta,
        A_s=args.A,
        Emin_kJ=args.emin,
        Emax_kJ=args.emax,
        nE=args.nE,
        Tgrid_C=(args.tmin, args.tmax, args.nT),
        lambda_smooth=args.lambda_smooth,
        w_tmax=args.w_tmax,
        w_hi=args.w_hi,
        fit_HI0=(args.fix_HI0 is None),
        HI0_fixed=args.fix_HI0,
        seed=args.seed
    )

    print("\n=== Inversion summary ===")
    print(f"Samples: {len(HI)}")
    print(f"β (°C/min): {args.beta:.3f}   A: {args.A:.2e} s^-1")
    print(f"E grid: {args.nE} from {args.emin:.1f} to {args.emax:.1f} kJ/mol")
    print(f"Fixed HI0: {fit.HI0:.1f} mg HC/g TOC")
    print(f"RMSE Tmax: {fit.rmse_Tmax:.2f} °C")
    print(f"RMSE HI/HI0: {fit.rmse_R:.4f}")

    # Print normalized HI fractions
    print("\nObserved HI vs. normalized fraction HI/HI0:")
    frac = HI / fit.HI0
    for i,(h,t,f_) in enumerate(zip(HI, Tmax, frac)):
        print(f" Sample {i+1}: HI={h:.2f}, Tmax={t:.1f} °C, HI/HI0={f_:.3f}")

    # Write CSVs
    out = df.copy()
    out["HI_pred"] = fit.R_pred * fit.HI0
    out["Tmax_pred_C"] = fit.T_pred_C
    out["maturity_scalar_m"] = fit.m
    out.to_csv("hi_tmax_fit_by_sample.csv", index=False)
    print("\nWrote per-sample results: hi_tmax_fit_by_sample.csv")

    f_df = pd.DataFrame({"E_kJmol": fit.E_kJmol, "f_E_density": fit.f})
    f_df.to_csv("fE_distribution.csv", index=False)
    print("Wrote kinetics distribution: fE_distribution.csv")

    E_mode = fit.E_kJmol[np.argmax(fit.f)]
    print(f"Mode of f(E): {E_mode:.1f} kJ/mol")

    # Plot HI/HI0 vs Tmax
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(Tmax, frac)
        plt.xlabel("Tmax (°C)")
        plt.ylabel("HI / HI0")
        plt.title("Normalized HI fraction vs. Tmax")
        plt.tight_layout()
        plt.savefig("HI_fraction_vs_Tmax.png", dpi=200)
        print("Wrote plot: HI_fraction_vs_Tmax.png")
        # plt.show()  # uncomment for interactive sessions
    except Exception as e:

        print(f"(Plot skipped: {e})")

def selftest():
    """Run a tiny test inversion with synthetic RockEval-like data."""
    import numpy as np
    import pandas as pd

    # Fake data: HI decreases with maturity, Tmax increases
    HI = np.array([550, 400, 250, 100], dtype=float)
    Tmax = np.array([430, 445, 460, 480], dtype=float)
    S1 = np.zeros_like(HI)
    S2 = np.zeros_like(HI)

    df = pd.DataFrame({"hi": HI, "tmax": Tmax, "s1": S1, "s2": S2})
    df.to_csv("synthetic_test.csv", index=False)

    fit = invert_hi_tmax(
        HI_obs=HI,
        Tmax_obs_C=Tmax,
        beta_C_per_min=25.0,
        A_s=1e14,
        HI0_fixed=600.0,
        fit_HI0=False,
    )

    print("\n=== Selftest summary ===")
    print(f"RMSE Tmax = {fit.rmse_Tmax:.2f} °C")
    print(f"RMSE HI/HI0 = {fit.rmse_R:.4f}")
    print("Mode of f(E) = %.1f kJ/mol" % fit.E_kJmol[np.argmax(fit.f)])

    # Write outputs like normal run
    out = df.copy()
    out["HI_pred"] = fit.R_pred * fit.HI0
    out["Tmax_pred_C"] = fit.T_pred_C
    out["maturity_scalar_m"] = fit.m
    out.to_csv("hi_tmax_fit_by_sample.csv", index=False)
    pd.DataFrame({"E_kJmol": fit.E_kJmol, "f_E_density": fit.f}).to_csv("fE_distribution.csv", index=False)
    print("Wrote synthetic outputs: hi_tmax_fit_by_sample.csv, fE_distribution.csv")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            main()
        except Exception as e:
            import traceback
            print("\n[ERROR] Exception while running:")
            traceback.print_exc()
            raise
