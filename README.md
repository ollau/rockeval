# RockEval kinetics inversion (HI + Tmax)

Dette repoet inneholder et Python-skript for å inverere kildebergarts-kinetikk  
fra rutine **Rock-Eval data** (HI, Tmax).  
Skriptet estimerer en aktiveringsenergi-fordeling `f(E)`, modellerer Tmax og HI/HI0,  
og skriver resultater til CSV + plott.

---

## 🔧 Avhengigheter

Installeres fra `requirements.txt`:

```bash
pip install -r requirements.txt

# How to use with your data

Export your S2 pyrogram as temperature (°C) vs. instantaneous S2 signal (e.g., mg HC/g/°C). If your instrument exports cumulative S2, differentiate numerically first.

Set the heating rate (commonly 25–30 °C/min for standard Rock-Eval).

Tune A_s if needed (1e14 s⁻¹ is a common assumption); adjust Emin_kJ, Emax_kJ, nE, and reg_lambda to stabilize/smooth 𝑓
(𝐸)f(E).Run the script; it prints modeled 𝑇max,Tmax, RMSE, and returns 𝑓(𝐸)f(E). The plotting block shows the fit and the recovered energy distribution.

Notes & extensions

If you only have routine Rock-Eval (S1, S2, HI, 𝑇max,Tmax) across a maturity series (no full pyrograms), you can adapt this by fitting model-predicted 𝑇max and HI across samples instead of the full rate curve (that’s essentially the idea in Chen et al., 2017; you would forward-model pyrograms for trial kinetics at each maturity proxy, then match observables).

ScienceDirect

Multi-ramp data (different 𝛽 β) further constrains the inversion; extend build_kernel per ramp and stack rows.

Keep 𝑓(𝐸)f(E) non-negative and apply mild smoothness (second-difference) to avoid spurious spikes—typical in ill-posed inversions. A modest reg_lambda (0.1–0.5) usually works well.

Key references (for method context & examples):

Chen, Z., Liu, X., Guo, Q., Jiang, C., & Mort, A. (2017). Inversion of source rock hydrocarbon generation kinetics from Rock-Eval data. Fuel, 194, 91–101. (Concept/motivation; routine Rock-Eval used for kinetic inversion.) 
ScienceDirect

Chen et al. (GeoConvention 2018): Forward and inverse modeling of kerogen generation kinetics based on routine Rock-Eval pyrolysis… (Helpful overview slides.) GeoConvention | Earth Science Conference

Government of Canada GEOSCAN metadata pointing to the 2017 Fuel paper. 
osdp-
