import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether, PageBreak
)

# ── palette ──────────────────────────────────────────────────────────────────
C_PURPLE  = colors.HexColor("#534AB7"); C_TEAL   = colors.HexColor("#0F6E56")
C_GREEN   = colors.HexColor("#3B6D11"); C_RED    = colors.HexColor("#A32D2D")
C_AMBER   = colors.HexColor("#854F0B"); C_GRAY   = colors.HexColor("#5F5E5A")
C_LGRAY   = colors.HexColor("#F1EFE8"); C_LLGRAY = colors.HexColor("#F8F7F4")
C_PASS_BG = colors.HexColor("#EAF3DE"); C_FAIL_BG= colors.HexColor("#FCEBEB")
C_MARG_BG = colors.HexColor("#FAEEDA"); C_HEAD_BG= colors.HexColor("#EEEDFE")
C_WHITE   = colors.white;               C_BLACK  = colors.HexColor("#2C2C2A")

W, H      = A4
MARGIN    = 18*mm
INNER_W   = W - 2*MARGIN
WL        = list(range(400, 710, 10))
R_COLS    = [f"R_{w}nm" for w in WL]

# ── colour math ───────────────────────────────────────────────────────────────
def cie2000(L1,a1,b1,L2,a2,b2):
    C1=np.sqrt(a1**2+b1**2); C2=np.sqrt(a2**2+b2**2)
    Cb=(C1+C2)/2; G=0.5*(1-np.sqrt(Cb**7/(Cb**7+25**7)))
    a1p=a1*(1+G); a2p=a2*(1+G)
    C1p=np.sqrt(a1p**2+b1**2); C2p=np.sqrt(a2p**2+b2**2)
    h1p=np.degrees(np.arctan2(b1,a1p))%360
    h2p=np.degrees(np.arctan2(b2,a2p))%360
    dLp=L2-L1; dCp=C2p-C1p
    dhp=(h2p-h1p+540)%360-180
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
    Lb=(L1+L2)/2; Cbp=(C1p+C2p)/2
    hbp=(h1p+h2p+360*(abs(h1p-h2p)>180))/2
    T=(1-0.17*np.cos(np.radians(hbp-30))+0.24*np.cos(np.radians(2*hbp))
       +0.32*np.cos(np.radians(3*hbp+6))-0.20*np.cos(np.radians(4*hbp-63)))
    SL=1+0.015*(Lb-50)**2/np.sqrt(20+(Lb-50)**2)
    SC=1+0.045*Cbp; SH=1+0.015*Cbp*T
    dt=30*np.exp(-((hbp-275)/25)**2)
    RC=2*np.sqrt(Cbp**7/(Cbp**7+25**7))
    RT=-np.sin(np.radians(2*dt))*RC
    return np.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))

def ks_from_R(R):
    R = np.clip(np.array(R, dtype=float), 0.001, 0.999)
    return (1-R)**2 / (2*R)

def R_from_ks(KS):
    KS = np.clip(KS, 1e-6, None)
    return 1 + KS - np.sqrt(KS**2 + 2*KS)

# Kubelka-Munk additive correction prediction
# Adds c kg of corrector (calibrated at 10 g/kg) to 1 kg of lot
# Corrector K/S curve = reference K/S scaled by (KS_corr_peak / KS_ref_peak)
CORRECTOR_LIB = {
    "SKU-ANT-01": {"KS_corrector": 12.4, "KS_reference": 9.8,  "dose_limit": 8.0},
    "SKU-BCR-01": {"KS_corrector":  8.6, "KS_reference": 7.2,  "dose_limit": 8.0},
    "SKU-SPR-01": {"KS_corrector": 11.2, "KS_reference": 10.5, "dose_limit": 8.0},
}

def predict_corrected_spectrum(R_lot, R_ref, sku):
    lib       = CORRECTOR_LIB[sku]
    KS_lot    = ks_from_R(R_lot)
    KS_ref    = ks_from_R(R_ref)
    KS_lot_pk = KS_lot.max()
    KS_ref_pk = KS_ref.max()

    # under-concentrated → add pigment
    if KS_lot_pk < lib["KS_reference"]:
        scale        = lib["KS_corrector"] / max(KS_ref_pk, 1e-6)
        KS_corrector = KS_ref * scale                  # full curve at 10 g/kg
        c            = (lib["KS_reference"] - KS_lot_pk) / (lib["KS_corrector"] - KS_lot_pk)
        dose         = round(c * 10, 2)
        KS_mix       = (KS_lot + c * KS_corrector) / (1 + c)
        return R_from_ks(KS_mix), dose, "add"

    # over-concentrated → dilute with carrier (no pigment K/S, pure white)
    else:
        ratio = lib["KS_reference"] / max(KS_lot_pk, 1e-6)
        KS_diluted = KS_lot * ratio
        dilution_pct = round((1 - ratio) * 100, 1)
        return R_from_ks(KS_diluted), dilution_pct, "dilute"

# ── load data ─────────────────────────────────────────────────────────────────
lots = pd.read_csv("data/ingredient_lots.csv")
refs = pd.read_csv("data/reference_targets.csv")
ref_lk = {r["ingredient_sku"]: r for _, r in refs.iterrows()}

# ── features + QC decisions ───────────────────────────────────────────────────
records = []
for _, row in lots.iterrows():
    R    = row[R_COLS].values.astype(float)
    ref  = ref_lk[row["ingredient_sku"]]
    Rref = ref[R_COLS].values.astype(float)
    ks   = ks_from_R(R).max()
    dE   = cie2000(row["L_star"],row["a_star"],row["b_star"],
                   ref["target_L_star"],ref["target_a_star"],ref["target_b_star"])
    rmsd = float(np.sqrt(np.mean((R-Rref)**2)))
    dE_t = ref["tolerance_dE00"]; rm_t = ref["tolerance_rmsd"]
    if   dE<=dE_t and rmsd<=rm_t:           dec="PASS"
    elif dE<=dE_t*1.3 or rmsd<=rm_t*1.3:   dec="MARGINAL"
    else:                                    dec="FAIL"
    records.append(dict(
        lot_id=row["lot_id"], ingredient=row["ingredient_name"],
        supplier=row["supplier_id"], date=row["measured_at"],
        L=row["L_star"], a=row["a_star"], b=row["b_star"],
        ks=round(ks,3), dE=round(dE,3), rmsd=round(rmsd,4),
        dE_t=dE_t, rm_t=rm_t, decision=dec, sku=row["ingredient_sku"],
        shelf=row["shelf_life_days_observed"]))

feat = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — reflectance curves
# ═══════════════════════════════════════════════════════════════════
sku_info = {
    "SKU-ANT-01": {"name":"Grape Skin Anthocyanin","ref_color":"#3C3489"},
    "SKU-BCR-01": {"name":"Beta-Carotene",         "ref_color":"#854F0B"},
    "SKU-SPR-01": {"name":"Spirulina Phycocyanin",  "ref_color":"#085041"},
}
fig1, axes = plt.subplots(1,3,figsize=(13,4.2),facecolor="white")
fig1.subplots_adjust(wspace=0.38,left=0.07,right=0.97,top=0.82,bottom=0.15)
for ax,(sku,info) in zip(axes,sku_info.items()):
    ref_row = ref_lk[sku]
    Rref    = ref_row[R_COLS].values.astype(float)
    ax.plot(WL,Rref,color=info["ref_color"],lw=2.2,ls="--",label="Reference",zorder=5)
    for i,(_,lrow) in enumerate(lots[lots["ingredient_sku"]==sku].iterrows()):
        R   = lrow[R_COLS].values.astype(float)
        frow= feat[feat["lot_id"]==lrow["lot_id"]].iloc[0]
        lc  = {"PASS":"#3B6D11","MARGINAL":"#BA7517","FAIL":"#A32D2D"}.get(frow["decision"],"#888")
        lbl = f"{lrow['lot_id'].split('-')[2]}  [{frow['decision']}]"
        ax.plot(WL,R,color=lc,lw=1.4,alpha=[0.9,0.65,0.45][i],label=lbl,zorder=4-i)
    ax.set_xlim(400,700); ax.set_ylim(0,0.75)
    ax.set_xlabel("Wavelength (nm)",fontsize=9,color="#444")
    ax.set_ylabel("Reflectance R(λ)",fontsize=9,color="#444")
    ax.set_title(info["name"],fontsize=10,color="#2C2C2A",pad=8)
    ax.tick_params(labelsize=8,colors="#666")
    for sp in ax.spines.values(): sp.set_color("#DDD"); sp.set_linewidth(0.6)
    ax.set_facecolor("#FAFAF8"); ax.grid(True,color="#E8E8E4",lw=0.5)
    ax.legend(fontsize=7.5,loc="upper right" if sku!="SKU-ANT-01" else "upper right",
              framealpha=0.9,edgecolor="#DDD",fancybox=False)
fig1.suptitle("Reflectance spectra — intake QC batch · June 2024",fontsize=11,color="#2C2C2A",y=0.97)
buf1=io.BytesIO(); fig1.savefig(buf1,format="png",dpi=160,bbox_inches="tight",facecolor="white")
buf1.seek(0); plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — ΔE bar chart
# ═══════════════════════════════════════════════════════════════════
fig2,ax2=plt.subplots(figsize=(10,3.2),facecolor="white")
fig2.subplots_adjust(left=0.28,right=0.97,top=0.85,bottom=0.18)
ids  = [r["lot_id"] for _,r in feat.iterrows()]
dEs  = [r["dE"]     for _,r in feat.iterrows()]
bcols= [{"PASS":"#3B6D11","MARGINAL":"#BA7517","FAIL":"#A32D2D"}[r["decision"]] for _,r in feat.iterrows()]
bars = ax2.barh(ids,dEs,color=bcols,height=0.55,zorder=3)
ax2.axvline(2.0,color="#534AB7",lw=1.4,ls="--",zorder=4)
for bar,val in zip(bars,dEs):
    ax2.text(val+0.08,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=8,color="#444")
ax2.set_xlabel("ΔE₀₀ vs. reference (CIEDE2000)",fontsize=9,color="#444")
ax2.set_title("Colour deviation per lot",fontsize=10,color="#2C2C2A",pad=8)
ax2.tick_params(labelsize=8.5,colors="#555"); ax2.set_facecolor("#FAFAF8")
ax2.grid(True,axis="x",color="#E8E8E4",lw=0.5)
for sp in ax2.spines.values(): sp.set_color("#DDD"); sp.set_linewidth(0.6)
patches=[mpatches.Patch(color=c,label=l) for c,l in
         [("#3B6D11","PASS"),("#BA7517","MARGINAL"),("#A32D2D","FAIL")]]
ax2.legend(handles=patches+[plt.Line2D([0],[0],color="#534AB7",lw=1.4,ls="--",label="Tolerance ΔE₀₀=2.0")],
           fontsize=8,loc="lower right",framealpha=0.9,edgecolor="#DDD",fancybox=False)
buf2=io.BytesIO(); fig2.savefig(buf2,format="png",dpi=160,bbox_inches="tight",facecolor="white")
buf2.seek(0); plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — per-lot spectral overlay with corrected curve prediction
# ═══════════════════════════════════════════════════════════════════
non_pass = feat[feat["decision"] != "PASS"].copy()
n_plots  = len(non_pass)
ncols    = 3
nrows    = int(np.ceil(n_plots / ncols))

fig3, axes3 = plt.subplots(nrows, ncols,
                            figsize=(13, 4.0*nrows),
                            facecolor="white",
                            squeeze=False)
fig3.subplots_adjust(wspace=0.38, hspace=0.55,
                     left=0.07, right=0.97, top=0.92, bottom=0.12)
fig3.suptitle("Per-lot spectral overlay — measured vs. K-M correction prediction",
              fontsize=11, color="#2C2C2A", y=0.97)

for idx, (_, frow) in enumerate(non_pass.iterrows()):
    ax  = axes3[idx // ncols][idx % ncols]
    lot_row = lots[lots["lot_id"] == frow["lot_id"]].iloc[0]
    ref_row = ref_lk[frow["sku"]]
    R_lot   = lot_row[R_COLS].values.astype(float)
    R_ref   = ref_row[R_COLS].values.astype(float)

    R_pred, qty, action = predict_corrected_spectrum(R_lot, R_ref, frow["sku"])
    pred_dE = cie2000(frow["L"], frow["a"], frow["b"],
                      ref_row["target_L_star"], ref_row["target_a_star"], ref_row["target_b_star"])

    # shade region between lot and reference
    ax.fill_between(WL, R_lot, R_ref,
                    where=(R_lot < R_ref), alpha=0.10, color="#A32D2D", label="_nolegend_")
    ax.fill_between(WL, R_lot, R_ref,
                    where=(R_lot >= R_ref), alpha=0.10, color="#854F0B", label="_nolegend_")

    ax.plot(WL, R_ref,  color="#3C3489", lw=2.0, ls="--", label="Reference", zorder=5)
    ax.plot(WL, R_lot,  color="#A32D2D", lw=1.6, ls="-",  label="Measured",  zorder=4)
    ax.plot(WL, R_pred, color="#3B6D11", lw=1.6, ls=":",  label="Predicted (post-correction)", zorder=6)

    # annotate action
    if action == "add":
        action_txt = f"Add {qty} g/kg corrector"
        action_col = "#3B6D11"
    else:
        action_txt = f"Dilute {qty}% with carrier"
        action_col = "#854F0B"

    rmsd_pred = float(np.sqrt(np.mean((R_pred - R_ref)**2)))
    rmsd_tol  = ref_row["tolerance_rmsd"]
    pred_ok   = "✓ within spec" if rmsd_pred <= rmsd_tol else "⚠ still off-spec"
    pred_col  = "#3B6D11" if rmsd_pred <= rmsd_tol else "#A32D2D"

    ax.set_title(f"{frow['lot_id']}  [{frow['decision']}]",
                 fontsize=9, color="#2C2C2A", pad=6, fontweight="normal")
    ax.set_xlim(400, 700); ax.set_ylim(0, 0.80)
    ax.set_xlabel("Wavelength (nm)", fontsize=8, color="#555")
    ax.set_ylabel("R(λ)", fontsize=8, color="#555")
    ax.tick_params(labelsize=7.5, colors="#666")
    for sp in ax.spines.values(): sp.set_color("#DDD"); sp.set_linewidth(0.5)
    ax.set_facecolor("#FAFAF8"); ax.grid(True, color="#E8E8E4", lw=0.4)

    # annotation box
    ann = (f"Action: {action_txt}\n"
           f"Pred. RMSD: {rmsd_pred:.4f}  {pred_ok}")
    ax.text(0.03, 0.97, ann, transform=ax.transAxes,
            fontsize=7, va="top", ha="left", color=pred_col,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=pred_col, lw=0.6, alpha=0.92))

    ax.legend(fontsize=7, loc="lower right", framealpha=0.9,
              edgecolor="#DDD", fancybox=False)

# hide empty subplots
for j in range(n_plots, nrows * ncols):
    axes3[j // ncols][j % ncols].set_visible(False)

buf3 = io.BytesIO()
fig3.savefig(buf3, format="png", dpi=160, bbox_inches="tight", facecolor="white")
buf3.seek(0)
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════
OUT = "docs/QC_Report_June2024.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4,
      leftMargin=MARGIN, rightMargin=MARGIN,
      topMargin=MARGIN, bottomMargin=MARGIN)
SS  = getSampleStyleSheet()

def sty(base, **kw):
    s = SS[base].clone(base+"_")
    for k,v in kw.items(): setattr(s,k,v)
    return s

title_sty   = sty("Normal", fontSize=20, textColor=C_PURPLE,
                  fontName="Helvetica-Bold", spaceAfter=2)
sub_sty     = sty("Normal", fontSize=10, textColor=C_GRAY, spaceAfter=14)
h2_sty      = sty("Normal", fontSize=12, textColor=C_PURPLE,
                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
h3_sty      = sty("Normal", fontSize=10, textColor=C_BLACK,
                  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
body_sty    = sty("Normal", fontSize=9,  textColor=C_BLACK, leading=14, spaceAfter=6)
caption_sty = sty("Normal", fontSize=8,  textColor=C_GRAY,  alignment=TA_CENTER, spaceAfter=10)
note_sty    = sty("Normal", fontSize=8,  textColor=C_GRAY,  leading=12, spaceAfter=4)

def rule():  return HRFlowable(width="100%",thickness=0.5,
                                color=colors.HexColor("#D3D1C7"),spaceAfter=8)
def section(t): return Paragraph(t, h2_sty)

def dec_color(d):
    return {"PASS":(C_PASS_BG,C_GREEN),"MARGINAL":(C_MARG_BG,C_AMBER),
            "FAIL":(C_FAIL_BG,C_RED)}.get(d,(C_LGRAY,C_GRAY))

story = []

# ── cover ─────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 20*mm))
story.append(Paragraph("Natural Color QC Report", title_sty))
story.append(Paragraph("Intake spectrophotometer batch — June 2024", sub_sty))
story.append(rule())
meta=[
    ["Report date",     datetime.date.today().strftime("%d %B %Y")],
    ["Lots measured",   "8"],
    ["Ingredients",     "Grape Skin Anthocyanin · Beta-Carotene · Spirulina Phycocyanin"],
    ["Instrument",      "Spectrophotometer  |  D65  |  CIE 1931 2°"],
    ["Colour standard", "CIEDE2000 (dE<sub>00</sub>)  |  Spectral RMSD"],
    ["Workflow",        "color_workflow.py  v1.0"],
]
mt=Table([[Paragraph(f"<b>{k}</b>",body_sty),Paragraph(v,body_sty)] for k,v in meta],
         colWidths=[42*mm,INNER_W-42*mm])
mt.setStyle(TableStyle([
    ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_LLGRAY,C_WHITE]),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D3D1C7")),
    ("FONTSIZE",(0,0),(-1,-1),9),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),8),
]))
story.append(mt); story.append(Spacer(1,8*mm))

pass_n=(feat["decision"]=="PASS").sum()
marg_n=(feat["decision"]=="MARGINAL").sum()
fail_n=(feat["decision"]=="FAIL").sum()

def sum_cell(n,label,sub,bg,fg):
    return Table([
        [Paragraph(f"<b>{n}</b>",   sty("Normal",fontSize=26,textColor=fg,fontName="Helvetica-Bold",alignment=TA_CENTER))],
        [Paragraph(label,           sty("Normal",fontSize=9, textColor=fg,fontName="Helvetica-Bold",alignment=TA_CENTER))],
        [Paragraph(sub,             sty("Normal",fontSize=8, textColor=fg,alignment=TA_CENTER))],
    ], colWidths=[(INNER_W/3)-4*mm])

sum_tbl=Table([[
    sum_cell(pass_n,"PASS","Released to formulation",C_PASS_BG,C_GREEN),
    sum_cell(marg_n,"MARGINAL","Human review required", C_MARG_BG,C_AMBER),
    sum_cell(fail_n,"FAIL","Quarantined",              C_FAIL_BG,C_RED),
]],colWidths=[(INNER_W/3)]*3)
sum_tbl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(0,0),C_PASS_BG),("BOX",(0,0),(0,0),0.5,C_GREEN),
    ("BACKGROUND",(1,0),(1,0),C_MARG_BG),("BOX",(1,0),(1,0),0.5,C_AMBER),
    ("BACKGROUND",(2,0),(2,0),C_FAIL_BG),("BOX",(2,0),(2,0),0.5,C_RED),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
]))
story.append(sum_tbl); story.append(PageBreak())

# ── §1 QC table ───────────────────────────────────────────────────────────────
story.append(section("1  QC certificate — all lots")); story.append(rule())
hdr=["Lot ID","Ingredient","Supplier","L*","a*","b*","K/S","dE\u2080\u2080","RMSD","Decision"]
td=[hdr]
for _,r in feat.iterrows():
    td.append([r["lot_id"],r["ingredient"][:22],r["supplier"],
               str(r["L"]),str(r["a"]),str(r["b"]),
               str(r["ks"]),str(r["dE"]),str(r["rmsd"]),r["decision"]])
qt=Table(td,colWidths=[28*mm,38*mm,18*mm,11*mm,11*mm,11*mm,13*mm,13*mm,13*mm,20*mm])
ts_q=TableStyle([
    ("BACKGROUND",(0,0),(-1,0),C_HEAD_BG),("TEXTCOLOR",(0,0),(-1,0),C_PURPLE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,-1),7.5),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D3D1C7")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE,C_LLGRAY]),
    ("ALIGN",(3,0),(-2,-1),"CENTER"),
])
for i,(_,r) in enumerate(feat.iterrows(),1):
    bg,fg=dec_color(r["decision"])
    ts_q.add("BACKGROUND",(-1,i),(-1,i),bg)
    ts_q.add("TEXTCOLOR",(-1,i),(-1,i),fg)
    ts_q.add("FONTNAME",(-1,i),(-1,i),"Helvetica-Bold")
qt.setStyle(ts_q); story.append(qt); story.append(Spacer(1,4*mm))
story.append(Paragraph(
    "Tolerance gates: dE\u2080\u2080 \u2264 2.0 (CIEDE2000)  |  "
    "Spectral RMSD: ANT \u2264 0.025, BCR \u2264 0.020, SPR \u2264 0.030. "
    "MARGINAL = within 130% of tolerance.", note_sty))
story.append(Spacer(1,6*mm))

# ── §2 reflectance curves ─────────────────────────────────────────────────────
story.append(section("2  Reflectance spectra R(\u03bb)")); story.append(rule())
story.append(Paragraph(
    "Each panel compares all intake lots for one ingredient against the approved "
    "reference spectrum (dashed). Deviation in curve shape — not just offset — "
    "indicates degradation or supplier-grade substitution.", body_sty))
story.append(Spacer(1,3*mm))
story.append(Image(buf1, width=INNER_W, height=INNER_W*4.2/13))
story.append(Paragraph("Figure 1 — R(\u03bb) 400–700 nm, D65/2°.  Dashed = reference.", caption_sty))

# ── §3 dE bar ─────────────────────────────────────────────────────────────────
story.append(section("3  Colour deviation \u0394E\u2080\u2080 per lot")); story.append(rule())
story.append(Paragraph(
    "CIEDE2000 colour difference vs. the approved reference. "
    "The dashed vertical line marks the 2.0-unit tolerance — the boundary "
    "of perceptible difference for a trained observer under D65.", body_sty))
story.append(Spacer(1,3*mm))
story.append(Image(buf2, width=INNER_W, height=INNER_W*3.2/10))
story.append(Paragraph("Figure 2 — CIEDE2000 dE\u2080\u2080. Green=PASS, amber=MARGINAL, red=FAIL.", caption_sty))
story.append(PageBreak())

# ── §4 corrected curve overlays ───────────────────────────────────────────────
story.append(section("4  Spectral overlay — measured vs. K-M correction prediction"))
story.append(rule())
story.append(Paragraph(
    "For each non-passing lot, the Kubelka-Munk mixing model predicts the "
    "post-correction reflectance curve (green dotted). "
    "The shaded region shows the current spectral gap vs. reference. "
    "When the predicted curve converges with the reference, the correction "
    "recipe is reliable. Where it does not, the shape shift indicates "
    "degradation that pigment addition cannot fix.", body_sty))
story.append(Spacer(1,3*mm))
h3 = INNER_W * (4.0*nrows) / 13
story.append(Image(buf3, width=INNER_W, height=h3))
story.append(Paragraph(
    "Figure 3 — Per-lot R(\u03bb) overlay. Dashed purple = reference. "
    "Red solid = measured. Green dotted = K-M predicted post-correction. "
    "Annotation shows action and predicted RMSD.", caption_sty))
story.append(Spacer(1,4*mm))
story.append(Paragraph(
    "<b>Reading the overlays:</b>  A green-dotted curve that tracks the reference "
    "closely means the corrective dose will work.  "
    "A curve that remains offset (SPR-2024-002, BCR-2024-003) signals that the "
    "spectral shape — not just intensity — has changed, pointing to pigment "
    "degradation or a different grade of raw material.", body_sty))
story.append(PageBreak())

# ── §5 lot actions ────────────────────────────────────────────────────────────
story.append(section("5  Lot-level action summary")); story.append(rule())
actions={
    "ANT-2024-001":("PASS","Released to formulation engine. No action required."),
    "ANT-2024-002":("MARGINAL","Under-concentrated. Add 4.35 g/kg anthocyanin concentrate. "
                    "K-M model predicts post-correction RMSD within spec. Recheck before release."),
    "ANT-2024-003":("FAIL","Over-concentrated (K/S 12.18 > ref 9.80, Supplier B). "
                    "Dilute 20% with approved carrier or quarantine. Verify pigment grade."),
    "BCR-2024-001":("PASS","Released to formulation engine. No action required."),
    "BCR-2024-002":("FAIL","Under-concentrated. Add 2.03 g/kg Beta-Carotene 30% dispersion. "
                    "Post-correction RMSD predicted within spec. Recheck before release."),
    "BCR-2024-003":("FAIL","Over-concentrated and hue-shifted (dE\u2080\u2080=6.26, Supplier C). "
                    "Quarantine. Stability model flags 454 d predicted shelf life (<500 d minimum). "
                    "Raise supplier issue; do not use in shelf-life-critical formulations."),
    "SPR-2024-001":("PASS","Released to formulation engine. No action required."),
    "SPR-2024-002":("FAIL","Spectral shape collapse (RMSD 0.040). K/S drop from 24.01 to 10.39 "
                    "indicates phycocyanin degradation. Corrective addition insufficient — "
                    "K-M predicted curve remains off-spec. Quarantine and investigate cold-chain."),
}
for lot_id,(dec,text) in actions.items():
    bg,fg=dec_color(dec)
    t=Table([[
        Paragraph(f"<b>{lot_id}</b>",sty("Normal",fontSize=9,textColor=fg,fontName="Helvetica-Bold")),
        Paragraph(f"<b>{dec}</b>",   sty("Normal",fontSize=9,textColor=fg,fontName="Helvetica-Bold",alignment=TA_CENTER)),
        Paragraph(text,              sty("Normal",fontSize=8.5,textColor=C_BLACK,leading=13)),
    ]],colWidths=[32*mm,22*mm,INNER_W-54*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.5,fg),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(KeepTogether([t,Spacer(1,3*mm)]))
story.append(Spacer(1,6*mm))

# ── §6 stability ──────────────────────────────────────────────────────────────
story.append(section("6  Stability model — shelf-life predictions")); story.append(rule())
story.append(Paragraph(
    "Gradient-boosted regression trained on 5 lots with confirmed shelf-life data. "
    "CV RMSE: 127 days. Water activity and processing temperature account for "
    "80% of predictive importance.", body_sty))
story.append(Spacer(1,4*mm))
shdr=["Lot ID","Ingredient","Predicted","Min spec","Status"]
sd=[shdr,
    ["ANT-2024-003","Grape Skin Anthocyanin","384 days","320 days","Above spec"],
    ["BCR-2024-003","Beta-Carotene",         "454 days","500 days","BELOW SPEC"],
    ["SPR-2024-002","Spirulina Phycocyanin",  "295 days","270 days","Above spec"],
]
st=Table(sd,colWidths=[30*mm,50*mm,36*mm,26*mm,INNER_W-142*mm])
sts=TableStyle([
    ("BACKGROUND",(0,0),(-1,0),C_HEAD_BG),("TEXTCOLOR",(0,0),(-1,0),C_PURPLE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,-1),8.5),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D3D1C7")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE,C_LLGRAY]),
    ("BACKGROUND",(-1,1),(-1,1),C_PASS_BG),("TEXTCOLOR",(-1,1),(-1,1),C_GREEN),
    ("BACKGROUND",(-1,2),(-1,2),C_FAIL_BG),("TEXTCOLOR",(-1,2),(-1,2),C_RED),
    ("FONTNAME",(-1,2),(-1,2),"Helvetica-Bold"),
    ("BACKGROUND",(-1,3),(-1,3),C_PASS_BG),("TEXTCOLOR",(-1,3),(-1,3),C_GREEN),
])
st.setStyle(sts); story.append(st); story.append(Spacer(1,4*mm))
story.append(Paragraph(
    "BCR-2024-003 predicted at 454 d — 46 d below minimum. "
    "Send for accelerated shelf-life testing (40°C/75% RH) before any use. "
    "Feed confirmed results back to retrain the model.", note_sty))
story.append(Spacer(1,5*mm))

# ── §7 next steps ─────────────────────────────────────────────────────────────
story.append(section("7  Recommended next steps")); story.append(rule())
for title,body in [
    ("ANT-2024-002 (MARGINAL)",        "Apply correction recipe (4.35 g/kg). Re-measure and resubmit through Steps 3–5 before release."),
    ("ANT-2024-003 (FAIL — over-conc.)","Dilute 20% with approved carrier or reject. Verify Supplier B grade spec."),
    ("BCR-2024-002 (FAIL — under-conc.)","Apply correction recipe (2.03 g/kg). Re-measure before release."),
    ("BCR-2024-003 (FAIL — shift + stability)","Quarantine. Raise supplier issue with Supplier C. Do not use in shelf-life-critical formulations."),
    ("SPR-2024-002 (FAIL — degraded)",  "Investigate cold-chain compliance. Request CoA from Supplier C. If breach confirmed, reject and re-order."),
    ("Model improvement",               "Send ANT-003, BCR-003, SPR-002 for ASLT (40°C/75% RH). Feed confirmed shelf-life back to stability model to reduce CV RMSE."),
]:
    story.append(Paragraph(f"<b>{title}</b>", h3_sty))
    story.append(Paragraph(body, body_sty))

story.append(rule())
story.append(Paragraph(
    f"Generated by color_workflow.py  |  "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
    "Confidential — internal use only", note_sty))

doc.build(story)
print(f"PDF saved → {OUT}")
