"""Natural Color Formulation AI — Streamlit Dashboard.

Deploy on Streamlit Community Cloud (free):
  1. Push this repo to GitHub
  2. Go to share.streamlit.io → New app → pick this repo
  3. Add GOOGLE_API_KEY to the app's Secrets (Settings → Secrets)
"""

import os, sys, re, datetime, io, warnings
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── password gate ─────────────────────────────────────────────────────────────
def _check_password() -> bool:
    correct = st.secrets.get("APP_PASSWORD", "")
    if not correct:
        return True  # no password configured → open access
    if st.session_state.get("authenticated"):
        return True
    st.title("🌿 Natural Color Formulation AI")
    pwd = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pwd == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

_check_password()


# ── pipeline math helpers ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_workflow import (
    _cie2000, _ks, _dominant_wavelength, _spectral_rmsd,
    CORRECTOR_LIBRARY, WAVELENGTHS, R_COLS,
)

# ── constants ──────────────────────────────────────────────────────────────────
MODEL      = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)
SKU_MAP = {
    "all":           None,
    "anthocyanin":   "SKU-ANT-01",
    "beta-carotene": "SKU-BCR-01",
    "phycocyanin":   "SKU-SPR-01",
}
SECTION_KEYS = [
    "Batch Overview",
    "Lot-by-Lot Analysis",
    "Corrective Formulation Recipes",
    "Shelf-Life Outlook",
    "Priority Action List",
]
SYSTEM_PROMPT = """\
You are an expert food colorist and plant pigment scientist. You have been given \
spectrophotometric QC data, Kubelka-Munk analysis results, and stability model \
outputs for a batch of natural food colorant lots.

Write a clear, expert formulation report with exactly these headings (## markdown):

## Batch Overview
## Lot-by-Lot Analysis
## Corrective Formulation Recipes
## Shelf-Life Outlook
## Priority Action List

Guidelines:
- Be precise and actionable. Cite specific lot IDs, dE values, doses, and days.
- Use bullet points for lists.
- For PASS lots: confirm fitness and note any marginal features worth monitoring.
- For FAIL/MARGINAL lots: explain the colour deviation and link it to the recipe.
- Priority Action List: rank lots by urgency (FAIL first, then MARGINAL, then PASS).
- Tailor language to the stated audience.
- Do not comment on the process of writing the report -- deliver findings only.
"""


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Natural Color Formulation AI",
    page_icon="🌿",
    layout="wide",
)


# ── API key resolution (server-side only — never sent to browser) ──────────────
def _get_api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")

api_key = _get_api_key()   # resolved once, stays on the server


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Natural Color AI")
    st.divider()

    st.divider()
    st.markdown("**Analysis Brief**")

    ingredient = st.selectbox(
        "Ingredient focus",
        ["all", "anthocyanin", "beta-carotene", "phycocyanin"],
        help="Filter lots by ingredient type",
    )
    focus = st.selectbox(
        "Primary focus",
        ["Full QC + formulation", "Corrective recipes only", "Shelf-life focus"],
    )
    audience = st.selectbox(
        "Report audience",
        ["Lab team", "Management", "Supplier"],
    )
    st.divider()
    st.markdown("**Data**")
    uploaded_lots = st.file_uploader(
        "Upload lots CSV (optional)",
        type="csv",
        help="Leave empty to use the built-in sample data. "
             "File must have the same columns as `data/ingredient_lots.csv`.",
    )
    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ── main header ───────────────────────────────────────────────────────────────
st.title("🌿 Natural Color Formulation AI")
st.caption(
    "Spectrophotometric QC · Kubelka-Munk correction · AI expert narrative  "
    f"| Model: {MODEL}"
)

if not run_btn:
    st.info("Choose your analysis brief in the sidebar, then click **Run Analysis**.")
    with st.expander("About this dashboard"):
        st.markdown("""
**Pipeline stages**
1. Load 8 sample lots from `data/ingredient_lots.csv`
2. Compute spectral features: K/S ratio, dominant wavelength, CIEDE2000 ΔE, RMSD
3. QC gate: PASS / MARGINAL / FAIL vs reference targets
4. Corrective recipe: pigment type + dose (g/kg) for non-passing lots
5. Stability model: gradient-boosted regression → predicted shelf-life days
6. AI narrative: Gemini Flash writes a 5-section expert formulation report

**Free hosting** · Deployed on [Streamlit Community Cloud](https://share.streamlit.io)
""")
    st.stop()


# ── validate key ──────────────────────────────────────────────────────────────
if not api_key:
    st.error(
        "API key not configured. "
        "Admin: add `GOOGLE_API_KEY` to Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ── analysis ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    base = os.path.dirname(os.path.abspath(__file__))
    if uploaded_file is not None:
        lots = pd.read_csv(uploaded_file)
    else:
        lots = pd.read_csv(os.path.join(base, "data", "ingredient_lots.csv"))
    refs = pd.read_csv(os.path.join(base, "data", "reference_targets.csv"))
    return lots, refs


def run_analysis(lots: pd.DataFrame, refs: pd.DataFrame) -> dict:
    ref_lookup = {r["ingredient_sku"]: r for _, r in refs.iterrows()}
    feat_rows = []
    for _, row in lots.iterrows():
        R     = row[R_COLS].values.astype(float)
        ref   = ref_lookup[row["ingredient_sku"]]
        R_ref = ref[R_COLS].values.astype(float)
        feat_rows.append(dict(
            lot_id         = row["lot_id"],
            ingredient_sku = row["ingredient_sku"],
            ingredient_name= row["ingredient_name"],
            supplier_id    = row["supplier_id"],
            measured_at    = row["measured_at"],
            L_star         = row["L_star"],
            a_star         = row["a_star"],
            b_star         = row["b_star"],
            pH             = row.get("pH", float("nan")),
            water_activity = row.get("water_activity", float("nan")),
            processing_temp= row.get("processing_temp_C", float("nan")),
            shelf_life_obs = row.get("shelf_life_days_observed", float("nan")),
            KS_peak        = round(float(_ks(R).max()), 4),
            dominant_wl_nm = _dominant_wavelength(R),
            dE00           = round(float(_cie2000(
                row["L_star"], row["a_star"], row["b_star"],
                ref["target_L_star"], ref["target_a_star"], ref["target_b_star"],
            )), 3),
            rmsd           = round(float(_spectral_rmsd(R, R_ref)), 4),
            chroma         = round(float(np.sqrt(row["a_star"]**2 + row["b_star"]**2)), 2),
            hue_angle      = round(float(np.degrees(np.arctan2(row["b_star"], row["a_star"])) % 360), 1),
            dE_limit       = float(ref["tolerance_dE00"]),
            rmsd_limit     = float(ref["tolerance_rmsd"]),
            min_shelf_life = float(ref["min_shelf_life_days"]),
        ))
    feat_df = pd.DataFrame(feat_rows)

    def _decide(row):
        dE_ok   = row["dE00"]  <= row["dE_limit"]
        rmsd_ok = row["rmsd"]  <= row["rmsd_limit"]
        marginal = (
            (row["dE_limit"]   < row["dE00"]  <= row["dE_limit"]  * 1.3) or
            (row["rmsd_limit"] < row["rmsd"]  <= row["rmsd_limit"] * 1.3)
        )
        if dE_ok and rmsd_ok: return "PASS"
        if marginal:          return "MARGINAL"
        return "FAIL"

    feat_df["decision"] = feat_df.apply(_decide, axis=1)

    recipes = []
    for _, f in feat_df[feat_df["decision"].isin(["FAIL", "MARGINAL"])].iterrows():
        lib = CORRECTOR_LIBRARY.get(f["ingredient_sku"])
        if not lib:
            continue
        KS_curr = f["KS_peak"]
        KS_ref  = lib["KS_reference"]
        KS_corr = lib["KS_corrector"]
        if KS_curr > KS_ref:
            recipes.append(dict(
                lot_id=f["lot_id"], action="DILUTE",
                corrector=lib["corrector"], dose_g_per_kg=None,
                exp_dE_post=None, confidence=None,
                note="Over-concentrated: dilute with carrier or reject",
            ))
        elif KS_corr > KS_curr:
            c    = (KS_ref - KS_curr) / (KS_corr - KS_curr)
            dose = round(c * 10, 2)
            exp  = round(f["dE00"] * (1 - c) * 0.6, 2)
            recipes.append(dict(
                lot_id=f["lot_id"], action="ADD",
                corrector=lib["corrector"], dose_g_per_kg=dose,
                exp_dE_post=exp,
                confidence="HIGH" if c < 0.5 else "MODERATE",
                note=None,
            ))

    merged   = feat_df.copy()
    merged["shelf_life_days_observed"] = pd.to_numeric(merged["shelf_life_obs"], errors="coerce")
    train    = merged[merged["shelf_life_days_observed"].notna()].copy()
    features = ["KS_peak", "rmsd", "pH", "water_activity", "processing_temp"]
    shelf_preds = {}
    cv_rmse = None
    if len(train) >= 3:
        X_train = train[features].fillna(train[features].mean()).values
        y_train = train["shelf_life_days_observed"].values
        gbr     = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
        gbr.fit(X_train, y_train)
        cv      = cross_val_score(gbr, X_train, y_train, cv=min(5, len(train)), scoring="neg_root_mean_squared_error")
        cv_rmse = round(float(-cv.mean()), 1)
        for (_, row), pred in zip(train.iterrows(), gbr.predict(X_train)):
            shelf_preds[row["lot_id"]] = round(float(row["shelf_life_days_observed"]), 0)
        predict = merged[merged["shelf_life_days_observed"].isna()].copy()
        if not predict.empty:
            X_pred = predict[features].fillna(train[features].mean()).values
            for (_, row), pred in zip(predict.iterrows(), gbr.predict(X_pred)):
                shelf_preds[row["lot_id"]] = round(float(pred), 0)

    return dict(feat_df=feat_df, recipes=recipes, shelf_preds=shelf_preds, cv_rmse=cv_rmse)


def build_prompt(brief, results):
    feat_df     = results["feat_df"]
    recipes     = results["recipes"]
    shelf_preds = results["shelf_preds"]
    cv_rmse     = results["cv_rmse"]
    recipe_map  = {r["lot_id"]: r for r in recipes}
    n_pass = (feat_df["decision"] == "PASS").sum()
    n_marg = (feat_df["decision"] == "MARGINAL").sum()
    n_fail = (feat_df["decision"] == "FAIL").sum()
    skus   = feat_df.groupby("ingredient_sku")["lot_id"].count().to_dict()
    lines  = [
        "=== BATCH DATA ===",
        f"Lots analysed: {len(feat_df)}  |  " + "  |  ".join(f"{k}: {v}" for k, v in skus.items()),
        f"QC Summary: PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}", "",
    ]
    for _, f in feat_df.iterrows():
        obs     = shelf_preds.get(f["lot_id"])
        obs_str = (
            f"predicted={obs:.0f} days" if pd.isna(f["shelf_life_obs"]) and obs
            else f"observed={f['shelf_life_obs']:.0f} days" if not pd.isna(f["shelf_life_obs"])
            else "unknown"
        )
        sl_flag = "OK" if obs and obs >= f["min_shelf_life"] else "BELOW MIN SPEC" if obs else ""
        lines += [
            f"[LOT: {f['lot_id']}  |  {f['ingredient_sku']}  |  {f['ingredient_name']}  |  Supplier: {f['supplier_id']}]",
            f"  Color: L*={f['L_star']}  a*={f['a_star']}  b*={f['b_star']}",
            f"  dE00={f['dE00']} (limit {f['dE_limit']})  |  RMSD={f['rmsd']} (limit {f['rmsd_limit']})  ->  {f['decision']}",
            f"  KS_peak={f['KS_peak']}  |  dom_wl={f['dominant_wl_nm']}nm  |  chroma={f['chroma']}  |  hue={f['hue_angle']}",
            f"  pH={f['pH']}  |  aw={f['water_activity']}  |  temp={f['processing_temp']}C",
            f"  Shelf-life: {obs_str}  (min spec: {f['min_shelf_life']:.0f} days)  {sl_flag}",
        ]
        if f["lot_id"] in recipe_map:
            r = recipe_map[f["lot_id"]]
            lines.append(
                f"  Recipe: Add {r['corrector']} @ {r['dose_g_per_kg']} g/kg  |  Est. dE00 post={r['exp_dE_post']}  |  {r['confidence']}"
                if r["action"] == "ADD" else f"  Recipe: {r['note']}"
            )
        lines.append("")
    if cv_rmse:
        lines += [f"Stability model CV RMSE: +/-{cv_rmse} days", ""]
    lines += [
        "=== BRIEF ===",
        f"Ingredient focus: {brief['ingredient']}",
        f"Primary focus: {brief['focus']}",
        f"Report audience: {brief['audience']}",
        "", "Write the expert formulation report using the specified headings.",
    ]
    return "\n".join(lines)


def call_gemini(prompt_block: str, key: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents":           [{"parts": [{"text": prompt_block}]}],
        "generationConfig":   {"maxOutputTokens": 8192},
    }
    resp = requests.post(GEMINI_URL, params={"key": key}, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def parse_sections(raw: str) -> dict:
    pattern = re.compile(
        r"##\s+(" + "|".join(re.escape(k) for k in SECTION_KEYS) + r")\s*\n",
        re.IGNORECASE,
    )
    parts = pattern.split(raw)
    sections: dict = {}
    i = 1
    while i < len(parts) - 1:
        sections[parts[i].strip()] = parts[i + 1].strip()
        i += 2
    return sections or {"Batch Overview": raw.strip()}


# ── run ───────────────────────────────────────────────────────────────────────
brief = {"ingredient": ingredient, "focus": focus, "audience": audience}

with st.spinner("Loading CSV data..."):
    try:
        lots, refs = load_data(uploaded_lots)
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

sku = SKU_MAP.get(ingredient)
if sku:
    lots = lots[lots["ingredient_sku"] == sku]
    if lots.empty:
        st.warning(f"No lots found for '{ingredient}'. Showing all lots.")
        lots, refs = load_data(None)

with st.spinner("Running spectral analysis pipeline..."):
    results = run_analysis(lots, refs)

feat_df = results["feat_df"]
n_pass  = int((feat_df["decision"] == "PASS").sum())
n_marg  = int((feat_df["decision"] == "MARGINAL").sum())
n_fail  = int((feat_df["decision"] == "FAIL").sum())

# ── metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lots analysed", len(feat_df))
c2.metric("PASS",     n_pass,  delta=None)
c3.metric("MARGINAL", n_marg)
c4.metric("FAIL",     n_fail)

st.divider()

# ── QC table ──────────────────────────────────────────────────────────────────
st.subheader("QC Certificate")

DECISION_COLORS = {"PASS": "background-color:#EAF3DE", "MARGINAL": "background-color:#FAEEDA", "FAIL": "background-color:#FCEBEB"}

def _color_decision(val):
    return DECISION_COLORS.get(val, "")

display = feat_df[[
    "lot_id", "ingredient_name", "supplier_id",
    "L_star", "a_star", "b_star", "dE00", "rmsd", "decision",
]].rename(columns={
    "lot_id": "Lot ID", "ingredient_name": "Ingredient", "supplier_id": "Supplier",
    "L_star": "L*", "a_star": "a*", "b_star": "b*", "dE00": "dE00", "rmsd": "RMSD",
    "decision": "Decision",
})
st.dataframe(
    display.style.map(_color_decision, subset=["Decision"]),
    use_container_width=True,
    hide_index=True,
)

# ── corrective recipes ────────────────────────────────────────────────────────
if results["recipes"]:
    st.subheader("Corrective Recipes")
    rec_df = pd.DataFrame(results["recipes"])[["lot_id", "action", "corrector", "dose_g_per_kg", "exp_dE_post", "confidence"]]
    rec_df.columns = ["Lot ID", "Action", "Corrector", "Dose (g/kg)", "Est. dE00 post", "Confidence"]
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.divider()

# ── AI narrative ──────────────────────────────────────────────────────────────
st.subheader("AI Expert Narrative")
with st.spinner(f"Sending data to {MODEL} for expert formulation narrative..."):
    try:
        prompt_block = build_prompt(brief, results)
        raw_text     = call_gemini(prompt_block, api_key)
        sections     = parse_sections(raw_text)
    except requests.HTTPError as e:
        st.error(f"Gemini API error: {e.response.status_code} — {e.response.text[:300]}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

tab_labels = [k for k in SECTION_KEYS if sections.get(k)]
tabs = st.tabs(tab_labels)
for tab, key in zip(tabs, tab_labels):
    with tab:
        st.markdown(sections[key])

st.divider()

# ── download ──────────────────────────────────────────────────────────────────
today = datetime.date.today().isoformat()
slug  = re.sub(r"[^a-z0-9]", "_", ingredient.lower())

md_parts = [
    f"# Color Formulation Report\n_Generated: {today} | Model: {MODEL}_\n",
    f"**Ingredient focus:** {ingredient}  \n**Primary focus:** {focus}  \n**Audience:** {audience}  \n",
    f"**Lots analysed:** {len(feat_df)}  |  PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}\n",
    "---\n",
]
for key in SECTION_KEYS:
    if sections.get(key):
        md_parts.append(f"## {key}\n\n{sections[key]}\n\n---\n")

st.download_button(
    label="Download Report (Markdown)",
    data="\n".join(md_parts).encode("utf-8"),
    file_name=f"formulation_report_{slug}_{today}.md",
    mime="text/markdown",
)
