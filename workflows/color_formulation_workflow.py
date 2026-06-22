#!/usr/bin/env python3
"""AI-powered color formulation workflow.

Reads ingredient_lots.csv + reference_targets.csv, runs the spectral
analysis pipeline, then sends all results to Claude which writes an
expert formulation narrative saved as MD + PDF.

Usage (run from the project root, i.e. the folder containing data/):
    python workflows/color_formulation_workflow.py

Requires: ANTHROPIC_API_KEY environment variable.
Outputs:  output/formulation_reports/<slug>_<date>.md  and  .pdf
"""

import datetime
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

import requests
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── import helpers from the main pipeline ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from color_workflow import (
    _cie2000, _ks, _dominant_wavelength, _spectral_rmsd,
    CORRECTOR_LIBRARY, WAVELENGTHS, R_COLS,
)

# ── constants ─────────────────────────────────────────────────────────────────
MODEL  = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)
W, H   = A4
MARGIN = 18 * mm

C_PURPLE = colors.HexColor("#534AB7")
C_TEAL   = colors.HexColor("#0F6E56")
C_BLACK  = colors.HexColor("#2C2C2A")
C_WHITE  = colors.white

SECTION_KEYS = [
    "Batch Overview",
    "Lot-by-Lot Analysis",
    "Corrective Formulation Recipes",
    "Shelf-Life Outlook",
    "Priority Action List",
]

SKU_MAP = {
    "all":          None,
    "anthocyanin":  "SKU-ANT-01",
    "beta-carotene": "SKU-BCR-01",
    "phycocyanin":  "SKU-SPR-01",
}

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
- Be precise and actionable. Cite specific lot IDs, ΔE values, doses, and days.
- Use bullet points for lists.
- For PASS lots: confirm fitness and note any marginal features worth monitoring.
- For FAIL/MARGINAL lots: explain the colour deviation and link it to the recipe.
- Priority Action List: rank lots by urgency (FAIL first, then MARGINAL, then PASS with caveats).
- Tailor language to the stated audience.
- Do not comment on the process of writing the report — deliver findings only.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — clarifying questions
# ─────────────────────────────────────────────────────────────────────────────
_QUESTIONS = [
    ("ingredient", "1. Which ingredient(s) to analyse?",
     "all / anthocyanin / beta-carotene / phycocyanin  (default: all)",
     True),
    ("focus",      "2. What is the primary focus?",
     "e.g. full QC + formulation / corrective recipes only / shelf-life focus",
     False),
    ("audience",   "3. Who is the report audience?",
     "e.g. lab team / management / supplier",
     False),
]


def _prompt(question: str, hint: str, optional: bool) -> str:
    suffix = " (press Enter for default)" if optional else ""
    print(f"\n  [{hint}]")
    while True:
        ans = input(f"  {question}{suffix}\n  > ").strip()
        if ans or optional:
            return ans
        print("  Please provide an answer.")


def clarify() -> dict:
    print("\n" + "=" * 58)
    print("  Color Formulation AI Workflow")
    print("=" * 58)
    print("  A few quick questions before analysis:\n")
    brief = {}
    for key, question, hint, optional in _QUESTIONS:
        brief[key] = _prompt(question, hint, optional)
    brief["ingredient"] = brief["ingredient"].strip().lower() or "all"
    return brief


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — run analysis pipeline, build structured prompt block
# ─────────────────────────────────────────────────────────────────────────────
def _load_data() -> tuple:
    lots_path = os.path.join("data", "ingredient_lots.csv")
    refs_path = os.path.join("data", "reference_targets.csv")
    if not os.path.exists(lots_path):
        sys.exit(
            f"\nERROR: Cannot find {lots_path}\n"
            "Run this script from the project root (the folder containing data/).\n"
        )
    lots = pd.read_csv(lots_path)
    refs = pd.read_csv(refs_path)
    return lots, refs


def _filter_lots(lots: pd.DataFrame, ingredient_key: str) -> pd.DataFrame:
    sku = SKU_MAP.get(ingredient_key)
    if sku:
        filtered = lots[lots["ingredient_sku"] == sku]
        if filtered.empty:
            print(f"  Warning: no lots found for '{ingredient_key}'. Using all lots.")
            return lots
        return filtered
    return lots


def _run_analysis(lots: pd.DataFrame, refs: pd.DataFrame) -> dict:
    ref_lookup = {r["ingredient_sku"]: r for _, r in refs.iterrows()}

    # ── feature extraction ──
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

    # ── QC decisions ──
    def _decide(row):
        dE_ok   = row["dE00"]  <= row["dE_limit"]
        rmsd_ok = row["rmsd"]  <= row["rmsd_limit"]
        marginal = (
            (row["dE_limit"]   < row["dE00"]  <= row["dE_limit"]  * 1.3) or
            (row["rmsd_limit"] < row["rmsd"]  <= row["rmsd_limit"] * 1.3)
        )
        if dE_ok and rmsd_ok:   return "PASS"
        if marginal:            return "MARGINAL"
        return "FAIL"

    feat_df["decision"] = feat_df.apply(_decide, axis=1)

    # ── corrective recipes ──
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
            c = (KS_ref - KS_curr) / (KS_corr - KS_curr)
            dose = round(c * 10, 2)
            exp_dE = round(f["dE00"] * (1 - c) * 0.6, 2)
            recipes.append(dict(
                lot_id=f["lot_id"], action="ADD",
                corrector=lib["corrector"], dose_g_per_kg=dose,
                exp_dE_post=exp_dE,
                confidence="HIGH" if c < 0.5 else "MODERATE",
                note=None,
            ))

    # ── stability model ──
    merged = feat_df.copy()
    merged["shelf_life_days_observed"] = pd.to_numeric(
        merged["shelf_life_obs"], errors="coerce"
    )
    train   = merged[merged["shelf_life_days_observed"].notna()].copy()
    predict = merged[merged["shelf_life_days_observed"].isna()].copy()
    features = ["KS_peak", "rmsd", "pH", "water_activity", "processing_temp"]

    shelf_preds = {}
    cv_rmse = None
    if len(train) >= 3:
        X_train = train[features].fillna(train[features].mean()).values
        y_train = train["shelf_life_days_observed"].values
        model   = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        cv = cross_val_score(
            model, X_train, y_train,
            cv=min(5, len(train)), scoring="neg_root_mean_squared_error"
        )
        cv_rmse = round(float(-cv.mean()), 1)

        # observed lots: in-sample prediction for context
        preds_train = model.predict(X_train)
        for (_, row), pred in zip(train.iterrows(), preds_train):
            shelf_preds[row["lot_id"]] = round(float(row["shelf_life_days_observed"]), 0)

        if not predict.empty:
            X_pred = predict[features].fillna(train[features].mean()).values
            preds  = model.predict(X_pred)
            for (_, row), pred in zip(predict.iterrows(), preds):
                shelf_preds[row["lot_id"]] = round(float(pred), 0)

    return dict(
        feat_df  = feat_df,
        recipes  = recipes,
        shelf_preds = shelf_preds,
        cv_rmse  = cv_rmse,
    )


def build_prompt_block(brief: dict, results: dict) -> str:
    feat_df     = results["feat_df"]
    recipes     = results["recipes"]
    shelf_preds = results["shelf_preds"]
    cv_rmse     = results["cv_rmse"]

    recipe_map = {r["lot_id"]: r for r in recipes}

    n_pass = (feat_df["decision"] == "PASS").sum()
    n_marg = (feat_df["decision"] == "MARGINAL").sum()
    n_fail = (feat_df["decision"] == "FAIL").sum()

    skus = feat_df.groupby("ingredient_sku")["lot_id"].count().to_dict()
    sku_summary = "  |  ".join(f"{k}: {v}" for k, v in skus.items())

    lines = [
        "=== BATCH DATA ===",
        f"Lots analysed: {len(feat_df)}  |  {sku_summary}",
        f"QC Summary: PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}",
        "",
    ]

    for _, f in feat_df.iterrows():
        obs = shelf_preds.get(f["lot_id"])
        obs_str = f"predicted={obs:.0f} days" if pd.isna(f["shelf_life_obs"]) and obs else \
                  f"observed={f['shelf_life_obs']:.0f} days" if not pd.isna(f["shelf_life_obs"]) else "unknown"
        min_sl  = f["min_shelf_life"]
        sl_flag = "✓" if obs and obs >= min_sl else "⚠ BELOW MIN SPEC" if obs else ""

        dE_flag   = "✓" if f["dE00"] <= f["dE_limit"] else "✗"
        rmsd_flag = "✓" if f["rmsd"] <= f["rmsd_limit"] else "✗"

        lines += [
            f"[LOT: {f['lot_id']}  |  {f['ingredient_sku']}  |  {f['ingredient_name']}  |  Supplier: {f['supplier_id']}  |  Measured: {f['measured_at']}]",
            f"  Color:   L*={f['L_star']}  a*={f['a_star']}  b*={f['b_star']}",
            f"  dE00={f['dE00']} (limit {f['dE_limit']}) {dE_flag}  |  RMSD={f['rmsd']} (limit {f['rmsd_limit']}) {rmsd_flag}  →  {f['decision']}",
            f"  KS_peak={f['KS_peak']}  |  dom_wl={f['dominant_wl_nm']}nm  |  chroma={f['chroma']}  |  hue={f['hue_angle']}°",
            f"  pH={f['pH']}  |  aw={f['water_activity']}  |  temp={f['processing_temp']}°C",
            f"  Shelf-life: {obs_str}  (min spec: {min_sl:.0f} days)  {sl_flag}",
        ]

        if f["lot_id"] in recipe_map:
            r = recipe_map[f["lot_id"]]
            if r["action"] == "ADD":
                lines.append(
                    f"  Recipe: Add {r['corrector']} @ {r['dose_g_per_kg']} g/kg  |  "
                    f"Est. post-correction dE00={r['exp_dE_post']}  |  Confidence: {r['confidence']}"
                )
            else:
                lines.append(f"  Recipe: {r['note']}")
        lines.append("")

    if cv_rmse is not None:
        lines += [f"Stability model CV RMSE: ±{cv_rmse} days", ""]

    lines += [
        "=== BRIEF ===",
        f"Ingredient focus: {brief['ingredient']}",
        f"Primary focus: {brief['focus']}",
        f"Report audience: {brief['audience']}",
        "",
        "Write the expert formulation report using the specified section headings.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Claude API call
# ─────────────────────────────────────────────────────────────────────────────
def generate_narrative(prompt_block: str) -> str:
    import time
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "\nERROR: GOOGLE_API_KEY is not set.\n"
            "  Get a free key at https://aistudio.google.com/\n"
            "  Then add  GOOGLE_API_KEY=your-key  to your .env file.\n"
        )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt_block}]}],
        "generationConfig": {"maxOutputTokens": 8192},
    }
    print("\n  Generating expert narrative", end="", flush=True)
    for attempt in range(4):
        resp = requests.post(
            GEMINI_URL, params={"key": api_key}, json=payload, timeout=120
        )
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"\n  Rate limit hit — waiting {wait}s (attempt {attempt+1}/4)...",
                  end="", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        print(" done.")
        return data["candidates"][0]["content"]["parts"][0]["text"]
    sys.exit(
        "\nERROR: Gemini API rate limit persists after 4 attempts.\n"
        "  Free tier allows 15 requests/minute. Wait 1 minute and retry.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — parse sections
# ─────────────────────────────────────────────────────────────────────────────
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
    if not sections:
        sections["Batch Overview"] = raw.strip()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5a — Markdown output
# ─────────────────────────────────────────────────────────────────────────────
def save_markdown(brief: dict, results: dict, sections: dict, path: str) -> None:
    feat_df = results["feat_df"]
    n_pass  = (feat_df["decision"] == "PASS").sum()
    n_marg  = (feat_df["decision"] == "MARGINAL").sum()
    n_fail  = (feat_df["decision"] == "FAIL").sum()
    today   = datetime.date.today().isoformat()

    lines = [
        "# Color Formulation Report",
        f"_Generated: {today}_",
        "",
        "---",
        "",
        "## Analysis Brief",
        f"- **Ingredient focus:** {brief['ingredient']}",
        f"- **Primary focus:** {brief['focus']}",
        f"- **Audience:** {brief['audience']}",
        f"- **Lots analysed:** {len(feat_df)}",
        f"- **QC summary:** PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}",
        "",
        "---",
        "",
    ]
    for key in SECTION_KEYS:
        body = sections.get(key, "")
        if body:
            lines += [f"## {key}", "", body, ""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5b — PDF output
# ─────────────────────────────────────────────────────────────────────────────
def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CFTitle", parent=base["Title"],
            fontSize=20, textColor=C_WHITE, spaceAfter=4,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "CFSub", parent=base["Normal"],
            fontSize=10, textColor=C_WHITE, spaceAfter=0,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "CFH2", parent=base["Heading2"],
            fontSize=12, textColor=C_TEAL,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "CFBody", parent=base["Normal"],
            fontSize=9.5, textColor=C_BLACK, leading=14,
            fontName="Helvetica", spaceAfter=5,
        ),
    }


def _md_to_paras(text: str, style) -> list:
    paras = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            line = "• " + line[2:]
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", line)
        line = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", line)
        paras.append(Paragraph(line, style))
    return paras


def save_pdf(brief: dict, results: dict, sections: dict, path: str) -> None:
    feat_df = results["feat_df"]
    n_pass  = (feat_df["decision"] == "PASS").sum()
    n_marg  = (feat_df["decision"] == "MARGINAL").sum()
    n_fail  = (feat_df["decision"] == "FAIL").sum()
    today   = datetime.date.today().isoformat()
    st      = _styles()
    story   = []

    banner = Table(
        [
            [Paragraph("Color Formulation Report", st["title"])],
            [Paragraph(f"Ingredient: {brief['ingredient'].title()}  ·  Focus: {brief['focus']}", st["subtitle"])],
            [Paragraph(f"Generated {today}  ·  {MODEL}", st["subtitle"])],
        ],
        colWidths=[W - 2 * MARGIN],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_TEAL),
        ("ROWPADDING",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, 0),  18),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 18),
    ]))
    story += [banner, Spacer(1, 8 * mm)]

    # brief box
    story.append(Paragraph("Analysis Brief", st["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_TEAL))
    story.append(Spacer(1, 3 * mm))
    for label, value in [
        ("Ingredient focus", brief["ingredient"]),
        ("Primary focus",    brief["focus"]),
        ("Audience",         brief["audience"]),
        ("Lots analysed",    str(len(feat_df))),
        ("QC summary",       f"PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}"),
    ]:
        story.append(Paragraph(f"<b>{label}:</b>  {value}", st["body"]))
    story.append(Spacer(1, 4 * mm))

    for key in SECTION_KEYS:
        body = sections.get(key, "")
        if not body:
            continue
        story.append(Paragraph(key, st["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_TEAL))
        story.append(Spacer(1, 2 * mm))
        story.extend(_md_to_paras(body, st["body"]))
        story.append(Spacer(1, 4 * mm))

    SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
    ).build(story)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def _slug(ingredient: str) -> str:
    s = re.sub(r"[^a-z0-9]", "_", ingredient.lower().strip())
    return re.sub(r"_+", "_", s).strip("_")[:30]


def run() -> None:
    brief = clarify()

    print("\n  Loading data and running analysis...")
    lots, refs = _load_data()
    lots = _filter_lots(lots, brief["ingredient"])
    results = _run_analysis(lots, refs)

    feat_df = results["feat_df"]
    n = len(feat_df)
    n_pass = (feat_df["decision"] == "PASS").sum()
    n_marg = (feat_df["decision"] == "MARGINAL").sum()
    n_fail = (feat_df["decision"] == "FAIL").sum()
    print(f"  {n} lots analysed  —  PASS={n_pass}  MARGINAL={n_marg}  FAIL={n_fail}")

    prompt_block = build_prompt_block(brief, results)
    raw_text     = generate_narrative(prompt_block)
    sections     = parse_sections(raw_text)

    today   = datetime.date.today().isoformat()
    out_dir = os.path.join("output", "formulation_reports")
    os.makedirs(out_dir, exist_ok=True)
    base    = os.path.join(out_dir, f"{_slug(brief['ingredient'])}_lots_{today}")

    print("\n  Saving outputs...")
    save_markdown(brief, results, sections, base + ".md")
    print(f"  Markdown -> {base}.md")
    save_pdf(brief, results, sections, base + ".pdf")
    print(f"  PDF      -> {base}.pdf")
    print("\n  Done.")


if __name__ == "__main__":
    run()
