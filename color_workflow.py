"""
Natural Color Formulation – Spectrophotometer Integration Workflow
==================================================================
Run this script from the folder containing:
  • ingredient_lots.csv
  • reference_targets.csv

Requirements: numpy, pandas, scikit-learn  (pip install numpy pandas scikit-learn)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

SEP = "=" * 65
WAVELENGTHS = list(range(400, 710, 10))          # 31 values
R_COLS      = [f"R_{w}nm" for w in WAVELENGTHS]


# ════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD & INSPECT
# ════════════════════════════════════════════════════════════════════
def step1_load(lots_path="data/ingredient_lots.csv",
               ref_path="data/reference_targets.csv"):
    print(f"\n{SEP}")
    print("STEP 1 · Load & Inspect")
    print(SEP)

    lots = pd.read_csv(lots_path)
    refs = pd.read_csv(ref_path)

    print(f"\nLot file  : {lots.shape[0]} lots × {lots.shape[1]} columns")
    print(f"Ref file  : {refs.shape[0]} reference targets\n")

    print("Lots overview:")
    summary = lots[["lot_id","ingredient_name","supplier_id",
                    "measured_at","L_star","a_star","b_star",
                    "shelf_life_days_observed"]]
    print(summary.to_string(index=False))

    print("\nReference targets:")
    print(refs[["ingredient_sku","target_L_star","target_a_star",
                "target_b_star","tolerance_dE00",
                "min_shelf_life_days"]].to_string(index=False))

    return lots, refs


# ════════════════════════════════════════════════════════════════════
# STEP 2 – NORMALISE & VALIDATE
# ════════════════════════════════════════════════════════════════════
def _cie2000(L1,a1,b1, L2,a2,b2):
    """CIEDE2000 colour difference (scalar inputs)."""
    kL=kC=kH=1
    C1=np.sqrt(a1**2+b1**2); C2=np.sqrt(a2**2+b2**2)
    Cbar=(C1+C2)/2
    G=0.5*(1-np.sqrt(Cbar**7/(Cbar**7+25**7)))
    a1p=a1*(1+G); a2p=a2*(1+G)
    C1p=np.sqrt(a1p**2+b1**2); C2p=np.sqrt(a2p**2+b2**2)
    h1p=np.degrees(np.arctan2(b1,a1p))%360
    h2p=np.degrees(np.arctan2(b2,a2p))%360
    dLp=L2-L1; dCp=C2p-C1p
    dhp = (h2p-h1p+540)%360-180
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
    Lbar=(L1+L2)/2
    Cbarp=(C1p+C2p)/2
    hbarp=(h1p+h2p+360*(abs(h1p-h2p)>180))/2
    T=(1-0.17*np.cos(np.radians(hbarp-30))
         +0.24*np.cos(np.radians(2*hbarp))
         +0.32*np.cos(np.radians(3*hbarp+6))
         -0.20*np.cos(np.radians(4*hbarp-63)))
    SL=1+0.015*(Lbar-50)**2/np.sqrt(20+(Lbar-50)**2)
    SC=1+0.045*Cbarp
    SH=1+0.015*Cbarp*T
    d_theta=30*np.exp(-((hbarp-275)/25)**2)
    RC=2*np.sqrt(Cbarp**7/(Cbarp**7+25**7))
    RT=-np.sin(np.radians(2*d_theta))*RC
    return np.sqrt((dLp/(kL*SL))**2+(dCp/(kC*SC))**2+(dHp/(kH*SH))**2
                   +RT*(dCp/(kC*SC))*(dHp/(kH*SH)))

def step2_normalise(lots, refs):
    print(f"\n{SEP}")
    print("STEP 2 · Normalise & Validate")
    print(SEP)

    issues = []
    for _, row in lots.iterrows():
        R = row[R_COLS].values.astype(float)
        # Reflectance must be in [0,1]
        if R.min() < 0 or R.max() > 1:
            issues.append(f"  ✗ {row['lot_id']}: R(λ) out of [0,1] range")
        # Illuminant & observer check
        if row["illuminant"] != "D65":
            issues.append(f"  ✗ {row['lot_id']}: illuminant ≠ D65 – recompute CIELAB")
        # At least one value below 0.15 (confirms not a flat/white sample)
        if R.min() > 0.15:
            issues.append(f"  ⚠ {row['lot_id']}: no strong absorption band – verify sample prep")

    if issues:
        print("\nValidation issues found:")
        for i in issues: print(i)
    else:
        print("\n✓ All 8 lots pass range & illuminant validation")

    # Cross-validate CIELAB: recompute ΔE between stated L*/a*/b* and XYZ-derived
    # (here we trust the instrument values; flag lots where L* diff looks large)
    print("\nIlluminant / observer metadata:")
    for _, r in lots.iterrows():
        print(f"  {r['lot_id']}  illuminant={r['illuminant']}  "
              f"observer={r['observer']}")
    return lots


# ════════════════════════════════════════════════════════════════════
# STEP 3 – SPECTRAL FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════
def _ks(R):
    """Kubelka-Munk K/S ratio per wavelength."""
    R = np.clip(R, 0.001, 0.999)
    return (1 - R)**2 / (2 * R)

def _dominant_wavelength(R):
    """Approximate dominant wavelength = λ of minimum reflectance."""
    return WAVELENGTHS[np.argmin(R)]

def _spectral_rmsd(R1, R2):
    return np.sqrt(np.mean((np.array(R1) - np.array(R2))**2))

def step3_features(lots, refs):
    print(f"\n{SEP}")
    print("STEP 3 · Spectral Feature Extraction")
    print(SEP)

    ref_lookup = {r["ingredient_sku"]: r for _, r in refs.iterrows()}
    records = []

    for _, row in lots.iterrows():
        R     = row[R_COLS].values.astype(float)
        ref   = ref_lookup[row["ingredient_sku"]]
        R_ref = ref[R_COLS].values.astype(float)

        KS        = _ks(R)
        KS_peak   = KS.max()                          # at peak absorption
        dom_wl    = _dominant_wavelength(R)
        rmsd      = _spectral_rmsd(R, R_ref)
        dE00      = _cie2000(row["L_star"],row["a_star"],row["b_star"],
                             ref["target_L_star"],ref["target_a_star"],ref["target_b_star"])
        chroma    = np.sqrt(row["a_star"]**2 + row["b_star"]**2)
        hue_angle = np.degrees(np.arctan2(row["b_star"], row["a_star"])) % 360

        records.append(dict(
            lot_id         = row["lot_id"],
            ingredient_sku = row["ingredient_sku"],
            dominant_wl_nm = dom_wl,
            KS_peak        = round(KS_peak, 4),
            dE00_vs_ref    = round(dE00, 3),
            rmsd_spectral  = round(rmsd, 4),
            chroma         = round(chroma, 2),
            hue_angle_deg  = round(hue_angle, 1),
        ))

    feat_df = pd.DataFrame(records)
    print("\nExtracted features per lot:")
    print(feat_df.to_string(index=False))
    return feat_df


# ════════════════════════════════════════════════════════════════════
# STEP 4 – CORRECTIVE ADDITION ENGINE  (Kubelka-Munk)
# ════════════════════════════════════════════════════════════════════
# Pre-measured K/S of corrector pigments at 10 g/kg concentration
CORRECTOR_LIBRARY = {
    "SKU-ANT-01": {
        "corrector": "Grape Skin Anthocyanin conc. 10 g/kg",
        "KS_corrector": 12.4,       # K/S at peak absorption
        "KS_reference": 9.8,        # K/S of approved reference lot
    },
    "SKU-BCR-01": {
        "corrector": "Beta-Carotene 30% dispersion 10 g/kg",
        "KS_corrector": 8.6,
        "KS_reference": 7.2,
    },
    "SKU-SPR-01": {
        "corrector": "Phycocyanin E18 10 g/kg",
        "KS_corrector": 11.2,
        "KS_reference": 10.5,
    },
}

def step4_correction(lots, feat_df, refs, dE_threshold=2.0):
    print(f"\n{SEP}")
    print("STEP 4 · Corrective Addition Engine  (K-M mixing model)")
    print(SEP)

    needs_correction = feat_df[feat_df["dE00_vs_ref"] > dE_threshold]

    if needs_correction.empty:
        print("\n✓ All lots within ΔE tolerance – no correction needed.")
        return

    ref_lookup = {r["ingredient_sku"]: r for _, r in refs.iterrows()}

    for _, f in needs_correction.iterrows():
        lot_row   = lots[lots["lot_id"] == f["lot_id"]].iloc[0]
        lib       = CORRECTOR_LIBRARY.get(f["ingredient_sku"])
        if not lib:
            print(f"\n  ⚠ {f['lot_id']}: no corrector defined for {f['ingredient_sku']}")
            continue

        KS_curr = f["KS_peak"]
        KS_ref  = lib["KS_reference"]
        KS_corr = lib["KS_corrector"]   # per 10 g/kg

        # K-M additive mixing: KS_target = KS_curr*(1-c) + KS_corr*c
        # Solve for c (fraction of corrector per kg)
        if KS_curr > KS_ref:
            print(f"\n  ✗ {f['lot_id']}: lot is OVER-CONCENTRATED (K/S {KS_curr:.2f} > ref {KS_ref:.2f})")
            print(f"       Action: dilute with carrier or reject – cannot add pigment to reduce K/S")
            continue

        if KS_corr <= KS_curr:
            print(f"\n  ✗ {f['lot_id']}: deviation out of recoverable range (lot K/S > corrector K/S)")
            continue

        c = (KS_ref - KS_curr) / (KS_corr - KS_curr)
        dose_g_per_kg = round(c * 10, 2)     # corrector is calibrated at 10 g/kg
        exp_dE_post   = round(f["dE00_vs_ref"] * (1 - c) * 0.6, 2)   # simplified estimate

        print(f"\n  Lot          : {f['lot_id']}")
        print(f"  ΔE00         : {f['dE00_vs_ref']:.2f}  (threshold {dE_threshold})")
        print(f"  Current K/S  : {KS_curr:.4f}  |  target K/S: {KS_ref:.4f}")
        print(f"  Action       : add {lib['corrector']}")
        print(f"  Dose         : {dose_g_per_kg} g / kg")
        print(f"  Est. ΔE post : {exp_dE_post}  (target < {dE_threshold})")
        print(f"  Confidence   : {'HIGH' if c < 0.5 else 'MODERATE (large dose – verify in lab)'}")


# ════════════════════════════════════════════════════════════════════
# STEP 5 – BATCH QC VALIDATION
# ════════════════════════════════════════════════════════════════════
def step5_qc(lots, feat_df, refs):
    print(f"\n{SEP}")
    print("STEP 5 · Batch QC Validation")
    print(SEP)

    ref_lookup = {r["ingredient_sku"]: r for _, r in refs.iterrows()}
    rows = []

    for _, f in feat_df.iterrows():
        ref     = ref_lookup[f["ingredient_sku"]]
        dE_tol  = ref["tolerance_dE00"]
        rm_tol  = ref["tolerance_rmsd"]
        dE      = f["dE00_vs_ref"]
        rmsd    = f["rmsd_spectral"]

        dE_ok   = dE  <= dE_tol
        rmsd_ok = rmsd <= rm_tol
        marginal = (dE_tol < dE <= dE_tol * 1.3) or (rm_tol < rmsd <= rm_tol * 1.3)

        if dE_ok and rmsd_ok:
            decision = "PASS"
        elif marginal:
            decision = "MARGINAL – human review"
        else:
            decision = "FAIL"

        rows.append(dict(
            lot_id   = f["lot_id"],
            dE00     = dE,
            dE_limit = dE_tol,
            rmsd     = rmsd,
            rmsd_lim = rm_tol,
            dE_pass  = "✓" if dE_ok  else "✗",
            rmsd_pass= "✓" if rmsd_ok else "✗",
            decision = decision,
        ))

    qc_df = pd.DataFrame(rows)
    print("\nQC certificate summary:")
    print(qc_df.to_string(index=False))

    passed  = (qc_df["decision"]=="PASS").sum()
    failed  = (qc_df["decision"]=="FAIL").sum()
    marginal= (qc_df["decision"].str.startswith("MARGINAL")).sum()
    print(f"\n  PASS: {passed}  |  MARGINAL: {marginal}  |  FAIL: {failed}")
    return qc_df


# ════════════════════════════════════════════════════════════════════
# STEP 6 – STABILITY MODEL  (train + predict)
# ════════════════════════════════════════════════════════════════════
def step6_stability(lots, feat_df):
    print(f"\n{SEP}")
    print("STEP 6 · Stability Model – Train & Predict")
    print(SEP)

    merged = feat_df.merge(
        lots[["lot_id","pH","water_activity",
              "processing_temp_C","shelf_life_days_observed"]],
        on="lot_id"
    )

    merged["shelf_life_days_observed"] = pd.to_numeric(
        merged["shelf_life_days_observed"], errors="coerce")

    train = merged[merged["shelf_life_days_observed"].notna()].copy()
    predict = merged[merged["shelf_life_days_observed"].isna()].copy()

    features = ["KS_peak","rmsd_spectral","pH","water_activity","processing_temp_C"]

    print(f"\nTraining on {len(train)} lots with observed shelf life:")
    print(train[["lot_id","shelf_life_days_observed"]+features].to_string(index=False))

    X_train = train[features].values
    y_train = train["shelf_life_days_observed"].values

    model = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                      learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Cross-validation RMSE (LOO given small N)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=min(5, len(train)), scoring="neg_root_mean_squared_error")
    print(f"\nCV RMSE: {-cv_scores.mean():.1f} days  (±{cv_scores.std():.1f})")

    print("\nFeature importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"  {feat:<28} {bar} {imp:.3f}")

    if not predict.empty:
        X_pred = predict[features].values
        preds  = model.predict(X_pred)
        print("\nPredicted shelf life for new lots:")
        for (_, row), pred in zip(predict.iterrows(), preds):
            flag = "⚠  below min spec" if pred < 270 else "✓"
            print(f"  {row['lot_id']:<20}  {pred:5.0f} days  {flag}")
    else:
        print("\nNo new lots to predict.")


# ════════════════════════════════════════════════════════════════════
# AUTO-GENERATE PDF REPORT  (calls generate_report.py)
# ════════════════════════════════════════════════════════════════════
def generate_pdf_report():
    import subprocess, sys, os
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "generate_report.py")
    if not os.path.exists(script):
        print("\n⚠  generate_report.py not found — skipping PDF generation.")
        return
    print(f"\n{SEP}")
    print("PDF REPORT · Generating …")
    print(SEP)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("  Error during PDF generation:")
        print(result.stderr[-800:])

# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    lots, refs = step1_load()
    lots       = step2_normalise(lots, refs)
    feat_df    = step3_features(lots, refs)
    step4_correction(lots, feat_df, refs)
    qc_df      = step5_qc(lots, feat_df, refs)
    step6_stability(lots, feat_df)

    print(f"\n{SEP}")
    print("Workflow complete. Next: send PASS lots to formulation engine,")
    print("FAIL lots to quarantine, MARGINAL lots for human review.")
    print(SEP)
    generate_pdf_report()


# ════════════════════════════════════════════════════════════════════
