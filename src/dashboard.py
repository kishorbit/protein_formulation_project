import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="FormulAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #05070d; color: #e2e8f0; }
.stApp { background-color: #05070d; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e1a 0%, #0d1120 100%); border-right: 1px solid #1e2535; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.card-header { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #6b748a; margin-bottom: 0.4rem; }
.card-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1.1; }
.card-sub { font-size: 0.75rem; color: #6b748a; margin-top: 0.3rem; }
.ood-warning { background: rgba(240,80,112,0.08); border: 1px solid rgba(240,80,112,0.35); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.ood-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #f05070; margin-bottom: 0.4rem; }
.ood-text { font-size: 0.8rem; color: #c07080; line-height: 1.6; }
.disclosure-box { background: rgba(240,160,32,0.08); border: 1px solid rgba(240,160,32,0.3); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; }
.disclosure-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #f0a020; margin-bottom: 0.6rem; }
.disclosure-text { font-size: 0.8rem; color: #c8a060; line-height: 1.7; }
.weight-total-ok  { color: #00d68a; font-weight: 700; font-size: 0.85rem; }
.weight-total-bad { color: #f05070; font-weight: 700; font-size: 0.85rem; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 4px; border: 1px solid #1e2535; }
.stTabs [data-baseweb="tab"] { color: #6b748a; font-size: 0.82rem; font-weight: 500; border-radius: 8px; padding: 6px 16px; }
.stTabs [aria-selected="true"] { background: rgba(0,200,240,0.1) !important; color: #00c8f0 !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#05070d","axes.facecolor":"#0a0e1a",
    "axes.edgecolor":"#1e2535","axes.labelcolor":"#6b748a",
    "xtick.color":"#6b748a","ytick.color":"#6b748a",
    "text.color":"#e2e8f0","grid.color":"#1e2535",
    "grid.linewidth":0.5,"font.family":"sans-serif","font.size":9,
})

CYAN="#00c8f0"; GREEN="#00d68a"; AMBER="#f0a020"
VIOLET="#7c6af7"; ROSE="#f05070"; MUTED="#6b748a"
TARGET_COLORS=[ROSE,AMBER,VIOLET,GREEN,CYAN]
TARGETS=["aggregation_score","oxidation_level","deamidation_level",
         "potency_retention","shelf_life_score"]
PRED=["pred_aggregation","pred_oxidation","pred_deamidation",
      "pred_potency","pred_shelf_life"]

@st.cache_data
def load_data():
    recs    = pd.read_csv("outputs/reports/excipient_recommendations.csv")
    pareto  = pd.read_csv("outputs/reports/pareto_recommendations.csv")
    merged  = pd.read_csv("data/processed/dataset_merged.csv")
    prot    = pd.read_csv("data/processed/protein_features.csv")
    val_df  = pd.read_csv("outputs/reports/stacked_validation.csv")
    base_df = pd.read_csv("outputs/reports/baseline_comparison.csv")
    sens_df = pd.read_csv("outputs/reports/weight_sensitivity.csv")
    ood_df  = pd.read_csv("outputs/reports/ood_report.csv")
    bias_df = pd.read_csv("outputs/reports/bias_corrections.csv")
    cls_df  = pd.read_csv("outputs/reports/class_auc_summary.csv")
    shap_path = "outputs/reports/shap_importance.csv"
    shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else pd.DataFrame()
    return recs,pareto,merged,prot,val_df,base_df,sens_df,ood_df,bias_df,cls_df,shap_df

recs,pareto,merged,prot_df,val_df,base_df,sens_df,ood_df,bias_df,cls_df,shap_df = load_data()

def derive_composite(df_in, w):
    return (w[0]*df_in["pred_aggregation"]
          + w[1]*df_in["pred_oxidation"]
          + w[2]*df_in["pred_deamidation"]
          + w[3]*(1-df_in["pred_potency"])
          + w[4]*(1-df_in["pred_shelf_life"]))

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:1rem 0 1.5rem;border-bottom:1px solid #1e2535;margin-bottom:1.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:#fff;">🧬 FormulAI</div>
        <div style="font-size:0.72rem;color:#6b748a;margin-top:2px;">Protein Formulation Intelligence</div>
    </div>""", unsafe_allow_html=True)

    proteins = sorted(merged["protein_id"].unique().tolist())
    sel_prot = st.selectbox("Select Protein", proteins)

    # OOD warning in sidebar
    ood_row = ood_df[ood_df["protein_id"]==sel_prot]
    if len(ood_row)>0:
        sev = ood_row.iloc[0]["ood_severity"]
        pct = ood_row.iloc[0]["pct_ood"]*100
        dist= ood_row.iloc[0]["mean_distance"]
        if sev in ["Medium","High"]:
            st.markdown(f"""<div class="ood-warning">
                <div class="ood-title">⚠ OOD Warning</div>
                <div class="ood-text">{pct:.0f}% of formulations fall outside
                training distribution.<br>Distance: {dist:.2f}<br>
                Predictions may be less reliable.</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="font-size:0.7rem;color:#6b748a;font-weight:600;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
    Composite Weights</div>""", unsafe_allow_html=True)
    st.caption("Weights must sum to 1.00.")

    w_agg  = st.slider("Aggregation",  0.0,1.0,0.35,0.05)
    w_ox   = st.slider("Oxidation",    0.0,1.0,0.25,0.05)
    w_deam = st.slider("Deamidation",  0.0,1.0,0.20,0.05)
    w_pot  = st.slider("Potency",      0.0,1.0,0.15,0.05)
    w_shelf= st.slider("Shelf life",   0.0,1.0,0.05,0.05)
    total  = round(w_agg+w_ox+w_deam+w_pot+w_shelf,2)

    if abs(total-1.0)<0.01:
        st.markdown(f'<div class="weight-total-ok">Total: {total:.2f} ✓</div>',
                    unsafe_allow_html=True)
        weights_valid=True
    else:
        st.markdown(f'<div class="weight-total-bad">Total: {total:.2f} — must equal 1.00</div>',
                    unsafe_allow_html=True)
        weights_valid=False

    if st.button("Reset to defaults"):
        st.rerun()
    weights=[w_agg,w_ox,w_deam,w_pot,w_shelf]

    st.markdown("---")
    st.markdown("""<div style="font-size:0.7rem;color:#6b748a;font-weight:600;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
    Stability Threshold</div>""", unsafe_allow_html=True)
    threshold = st.slider("Stable if score below:", 0.30,0.60,0.40,0.01)
    st.caption(f"Current: {threshold:.2f}  |  Default: 0.40")

    st.markdown("---")
    valid_auc = val_df["roc_auc"].dropna()
    n_nan     = val_df["roc_auc"].isna().sum()
    n_ood     = (ood_df["ood_severity"].isin(["Medium","High"])).sum()
    st.markdown(f"""<div style="display:flex;flex-direction:column;gap:0.6rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Proteins</span><span style="color:#e2e8f0;font-weight:600;">{len(proteins)}</span></div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Mean ROC-AUC</span><span style="color:#00d68a;font-weight:600;">{valid_auc.mean():.3f}</span></div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Min ROC-AUC</span><span style="color:#00c8f0;font-weight:600;">{valid_auc.min():.3f}</span></div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">OOD proteins</span><span style="color:#f05070;font-weight:600;">{n_ood}/40</span></div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Inherently stable</span><span style="color:#f0a020;font-weight:600;">{n_nan}/40</span></div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:6px 0;">
            <span style="color:#6b748a;">Architecture</span><span style="color:#7c6af7;font-weight:600;">Stacked</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("UniProt · PubChem · FDA DailyMed · PubMed")

# ── Recompute with user weights ───────────────────────────
if weights_valid:
    recs_live = recs.copy()
    recs_live["pred_composite_score"] = derive_composite(recs_live,weights).round(4)
    recs_live["pred_stable"] = (recs_live["pred_composite_score"]<threshold).astype(int)
    recs_live = recs_live.sort_values("pred_composite_score")
else:
    recs_live = recs.copy()

# ── Tabs ──────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "Recommendations","Pareto Front","SHAP",
    "Protein Profile","Validation","Residual Analysis",
    "Model Comparison","About & Scope"
])

# ── TAB 1 — Recommendations ──────────────────────────────
with tab1:
    if not weights_valid:
        st.warning("Weights do not sum to 1.00 — adjust sliders.")

    # OOD banner
    if len(ood_row)>0 and ood_row.iloc[0]["ood_severity"] in ["Medium","High"]:
        st.markdown(f"""<div class="ood-warning">
            <div class="ood-title">⚠ Out-of-Distribution Warning — {sel_prot}</div>
            <div class="ood-text">
            This protein sits outside the training distribution
            (Mahalanobis distance={ood_row.iloc[0]['mean_distance']:.2f},
            {ood_row.iloc[0]['pct_ood']*100:.0f}% of formulations flagged).
            Recommendations should be treated with extra caution and
            prioritised for experimental validation before use.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00c8f0;"></div>
        <div style="font-size:1rem;font-weight:600;">Excipient Recommendations</div>
    </div>""", unsafe_allow_html=True)

    sub = recs_live[recs_live["protein_id"]==sel_prot].copy()
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card"><div class="card-header">Best composite score</div>
            <div class="card-value" style="color:#00d68a;">{sub["pred_composite_score"].min():.3f}</div>
            <div class="card-sub">Lower = more stable</div></div>""",unsafe_allow_html=True)
    with c2:
        n_high=(sub.get("confidence","")=="High").sum() if "confidence" in sub.columns else 0
        st.markdown(f"""<div class="card"><div class="card-header">High confidence</div>
            <div class="card-value" style="color:#00c8f0;">{n_high}</div>
            <div class="card-sub">All 3 models agree</div></div>""",unsafe_allow_html=True)
    with c3:
        n_stab=int(sub["pred_stable"].sum()) if "pred_stable" in sub.columns else 0
        st.markdown(f"""<div class="card"><div class="card-header">Stable at {threshold:.2f}</div>
            <div class="card-value" style="color:#7c6af7;">{n_stab}</div>
            <div class="card-sub">of {len(sub)} formulations</div></div>""",unsafe_allow_html=True)
    with c4:
        best_agg=sub["pred_aggregation"].min() if "pred_aggregation" in sub.columns else 0
        st.markdown(f"""<div class="card"><div class="card-header">Best aggregation</div>
            <div class="card-value" style="color:#f0a020;">{best_agg:.3f}</div>
            <div class="card-sub">Predicted score</div></div>""",unsafe_allow_html=True)

    st.markdown("##### Top 20 recommendations")
    show_cols=[c for c in ["buffer","sugar","surfactant","amino_acid","ph",
        "temperature_c","pred_composite_score","pred_aggregation","pred_oxidation",
        "pred_deamidation","pred_potency","confidence"] if c in sub.columns]

    def style_conf(val):
        if val=="High":   return "color:#00d68a;font-weight:600"
        if val=="Medium": return "color:#f0a020;font-weight:600"
        if val=="Low":    return "color:#f05070;font-weight:600"
        return ""

    display=sub[show_cols].head(20).reset_index(drop=True)
    if "confidence" in display.columns:
        styled=display.style.applymap(style_conf,subset=["confidence"])\
            .format({c:"{:.3f}" for c in display.select_dtypes("float").columns})
        st.dataframe(styled,use_container_width=True,height=420)
    else:
        st.dataframe(display,use_container_width=True,height=420)

    if weights!=[0.35,0.25,0.20,0.15,0.05]:
        st.info(f"Custom weights active: Agg={w_agg} Ox={w_ox} "
                f"Deam={w_deam} Pot={w_pot} Shelf={w_shelf} | "
                f"Threshold={threshold:.2f}")

# ── TAB 2 — Pareto ────────────────────────────────────────
with tab2:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#7c6af7;"></div>
        <div style="font-size:1rem;font-weight:600;">Pareto-Optimal Formulations</div>
    </div>""", unsafe_allow_html=True)
    psub=pareto[pareto["protein_id"]==sel_prot].copy()
    if len(psub)==0:
        st.info("No Pareto-optimal formulations for this protein.")
    else:
        c1,c2,c3=st.columns(3)
        c1.markdown(f"""<div class="card"><div class="card-header">Pareto formulations</div>
            <div class="card-value" style="color:#7c6af7;">{len(psub)}</div>
            <div class="card-sub">Non-dominated solutions</div></div>""",unsafe_allow_html=True)
        c2.markdown(f"""<div class="card"><div class="card-header">Best composite</div>
            <div class="card-value" style="color:#00d68a;">
            {psub["composite_stability_score"].min():.4f}</div>
            <div class="card-sub">Lowest degradation risk</div></div>""",unsafe_allow_html=True)
        n_hp=(psub.get("confidence","")=="High").sum()
        c3.markdown(f"""<div class="card"><div class="card-header">High confidence</div>
            <div class="card-value" style="color:#00c8f0;">{n_hp}</div>
            <div class="card-sub">Pareto + all models agree</div></div>""",unsafe_allow_html=True)
        show_p=[c for c in ["buffer","sugar","surfactant","amino_acid","ph",
            "composite_stability_score","stability_score","low_aggregation",
            "low_oxidation","high_potency","confidence"] if c in psub.columns]
        st.dataframe(psub[show_p].reset_index(drop=True),
                     use_container_width=True,height=300)

# ── TAB 3 — SHAP ─────────────────────────────────────────
with tab3:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#f0a020;"></div>
        <div style="font-size:1rem;font-weight:600;">Feature Importance (SHAP)</div>
    </div>""", unsafe_allow_html=True)
    if shap_df.empty:
        st.warning("SHAP file not found — run shap analysis to populate this tab.")
    else:
        top_n=st.slider("Top N features",5,30,15)
        top=shap_df.head(top_n)
        fig,ax=plt.subplots(figsize=(9,top_n*0.42+1))
        colors=[CYAN if i<5 else GREEN if i<10 else MUTED for i in range(len(top))]
        ax.barh(top["feature"][::-1],top["shap_importance"][::-1],
                color=colors[::-1],alpha=0.85,height=0.65,edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|",color=MUTED)
        ax.grid(axis="x",alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 4 — Protein Profile ───────────────────────────────
with tab4:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00d68a;"></div>
        <div style="font-size:1rem;font-weight:600;">Protein Physicochemical Profile</div>
    </div>""", unsafe_allow_html=True)

    # OOD detail for this protein
    if len(ood_row)>0:
        sev=ood_row.iloc[0]["ood_severity"]
        dist=ood_row.iloc[0]["mean_distance"]
        pct=ood_row.iloc[0]["pct_ood"]*100
        color_map={"In-distribution":GREEN,"Medium":AMBER,"High":ROSE}
        color=color_map.get(sev,MUTED)
        st.markdown(f"""<div class="card">
            <div class="card-header">Distribution status</div>
            <div class="card-value" style="color:{color};font-size:1.2rem;">{sev}</div>
            <div class="card-sub">Mahalanobis distance: {dist:.2f} | {pct:.0f}% formulations OOD</div>
        </div>""", unsafe_allow_html=True)

    matches=prot_df[prot_df["protein_id"]==sel_prot]
    if len(matches)==0:
        st.warning("No profile found.")
    else:
        prow=matches.iloc[0]
        c1,c2,c3,c4=st.columns(4)
        for col,label,key,color in [
            (c1,"Isoelectric point","isoelectric_point",CYAN),
            (c2,"GRAVY score","gravy_score",GREEN),
            (c3,"Instability index","instability_index",AMBER),
            (c4,"Agg mean score","agg_mean",ROSE)]:
            val=prow.get(key,"N/A")
            val_str=f"{val:.3f}" if isinstance(val,float) else str(val)
            col.markdown(f"""<div class="card">
                <div class="card-header">{label}</div>
                <div class="card-value" style="color:{color};font-size:1.6rem;">{val_str}</div>
            </div>""",unsafe_allow_html=True)
        risk={l:prow[aa] for aa,l in [
            ("pct_asn","Asn (deamidation)"),("pct_met","Met (oxidation)"),
            ("pct_cys","Cys (disulfide)"),("pct_trp","Trp (oxidation)"),
            ("pct_his","His (pH-sensitive)")]
            if aa in prow.index and not pd.isna(prow[aa])}
        if risk:
            st.markdown("##### Degradation-risk residue content")
            fig2,ax2=plt.subplots(figsize=(7,2.8))
            ax2.bar(risk.keys(),risk.values(),
                    color=TARGET_COLORS[:len(risk)],alpha=0.85,width=0.5,edgecolor="none")
            ax2.set_ylabel("% of sequence",color=MUTED)
            ax2.grid(axis="y",alpha=0.3)
            ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── TAB 5 — Validation ────────────────────────────────────
with tab5:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#f05070;"></div>
        <div style="font-size:1rem;font-weight:600;">Held-Out Validation — Stacked Ensemble</div>
    </div>""", unsafe_allow_html=True)
    valid=val_df[val_df["roc_auc"].notna()]
    n_nan=val_df["roc_auc"].isna().sum()
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(f"""<div class="card"><div class="card-header">Mean ROC-AUC</div>
        <div class="card-value" style="color:#00d68a;">{valid["roc_auc"].mean():.3f}</div>
        <div class="card-sub">{len(valid)} proteins evaluated</div></div>""",unsafe_allow_html=True)
    c2.markdown(f"""<div class="card"><div class="card-header">Min ROC-AUC</div>
        <div class="card-value" style="color:#00c8f0;">{valid["roc_auc"].min():.3f}</div>
        <div class="card-sub">Worst-case protein</div></div>""",unsafe_allow_html=True)
    c3.markdown(f"""<div class="card"><div class="card-header">Mean composite R²</div>
        <div class="card-value" style="color:#7c6af7;">{val_df["composite_r2"].mean():.3f}</div>
        <div class="card-sub">All 40 proteins</div></div>""",unsafe_allow_html=True)
    c4.markdown(f"""<div class="card"><div class="card-header">Inherently stable</div>
        <div class="card-value" style="color:#f0a020;">{n_nan}</div>
        <div class="card-sub">Single-class proteins</div></div>""",unsafe_allow_html=True)

    fig3,ax3=plt.subplots(figsize=(12,4))
    auc_vals=val_df["roc_auc"].fillna(0).values
    bar_colors=[]
    for v,pid in zip(auc_vals,val_df["held_out_protein"]):
        if val_df[val_df["held_out_protein"]==pid]["roc_auc"].isna().values[0]:
            bar_colors.append(AMBER)
        elif v>=0.97: bar_colors.append(GREEN)
        elif v>=0.90: bar_colors.append(CYAN)
        else:         bar_colors.append(ROSE)
    ax3.bar(range(len(val_df)),auc_vals,color=bar_colors,alpha=0.85,width=0.7,edgecolor="none")
    ax3.axhline(0.97,color=GREEN,ls="--",lw=1,alpha=0.6,label="0.97")
    ax3.axhline(0.90,color=CYAN, ls="--",lw=1,alpha=0.6,label="0.90")
    ax3.set_xticks(range(len(val_df)))
    ax3.set_xticklabels(val_df["held_out_protein"],rotation=90,fontsize=6.5)
    ax3.set_ylabel("ROC-AUC"); ax3.set_ylim(0,1.08)
    ax3.set_title("Per-protein LOO ROC-AUC  (amber = inherently stable)",
                  color="#e2e8f0",pad=10)
    ax3.legend(fontsize=8,framealpha=0); ax3.grid(axis="y",alpha=0.3)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("##### Per-class performance")
    c1,c2,c3,c4=st.columns(4)
    colors_cls=[GREEN,GREEN,CYAN,GREEN]
    for col,((_,row),clr) in zip([c1,c2,c3,c4],
                                   zip(cls_df.iterrows(),colors_cls)):
        cls_name=str(row["class"]).split()[0].capitalize()
        col.markdown(f"""<div class="card">
            <div class="card-header">{cls_name}</div>
            <div class="card-value" style="color:{clr};font-size:1.4rem;">
            {row["mean_roc_auc"]:.3f}</div>
            <div class="card-sub">N={int(row["n_proteins"])} | min={row["min_roc_auc"]:.3f}</div>
        </div>""",unsafe_allow_html=True)

    st.dataframe(val_df,use_container_width=True,height=300)

# ── TAB 6 — Residual Analysis ────────────────────────────
with tab6:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00c8f0;"></div>
        <div style="font-size:1rem;font-weight:600;">Residual Analysis & Bias Correction</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### Bias corrections applied (LOO-estimated)")
    c_cols=st.columns(5)
    for col,(_,row) in zip(c_cols,bias_df.iterrows()):
        label=row["target"].replace("_score","").replace("_"," ").title()
        color=GREEN if abs(row["bias"])<0.002 else AMBER
        col.markdown(f"""<div class="card">
            <div class="card-header">{label}</div>
            <div class="card-value" style="color:{color};font-size:1.2rem;">
            {row["bias"]:+.4f}</div>
            <div class="card-sub">Bias corrected</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("##### Per-target LOO R² — known limitations")
    r2_cols=["r2_aggregation_score","r2_oxidation_level","r2_deamidation_level",
             "r2_potency_retention","r2_shelf_life_score"]
    r2_cols=[c for c in r2_cols if c in val_df.columns]
    if r2_cols:
        fig4,axes4=plt.subplots(1,len(r2_cols),figsize=(16,3.5))
        for ax,col,color,target in zip(axes4,r2_cols,TARGET_COLORS,
            ["Aggregation","Oxidation","Deamidation","Potency","Shelf life"]):
            vals=val_df[col].values
            ax.hist(vals,bins=15,color=color,alpha=0.8,edgecolor="none")
            ax.axvline(vals.mean(),color="white",lw=1.5,ls="--",
                       label=f"mean={vals.mean():.3f}")
            ax.axvline(0,color=ROSE,lw=1,ls=":",alpha=0.6)
            ax.set_title(target,fontsize=9,pad=6)
            ax.set_xlabel("LOO R²",fontsize=7)
            ax.legend(fontsize=7,framealpha=0)
            ax.grid(axis="y",alpha=0.3)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.suptitle("Per-protein LOO R² distributions",
                     fontsize=10,y=1.02,color="#e2e8f0")
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("""<div class="disclosure-box">
        <div class="disclosure-title">Oxidation model limitation</div>
        <div class="disclosure-text">
        Oxidation LOO R²=0.594 is the weakest target. SASA proxy features were tested
        and rolled back — they degraded all targets due to multicollinearity.
        True improvement requires AlphaFold-derived solvent-accessible surface area
        coordinates for Met and Trp residues. This is documented as a known limitation.
        </div>
    </div>""",unsafe_allow_html=True)

    st.markdown("##### OOD detection — protein distribution status")
    fig5,ax5=plt.subplots(figsize=(12,3.5))
    ood_sorted=ood_df.sort_values("mean_distance",ascending=False)
    bar_c=[ROSE if s=="High" else AMBER if s=="Medium" else GREEN
           for s in ood_sorted["ood_severity"]]
    ax5.barh(ood_sorted["protein_id"],ood_sorted["mean_distance"],
             color=bar_c,alpha=0.85,height=0.6,edgecolor="none")
    ax5.axvline(ood_df["mean_distance"].quantile(0.95),
                color=AMBER,ls="--",lw=1,label="95th percentile")
    ax5.set_xlabel("Mean Mahalanobis distance")
    ax5.set_title("Protein OOD scores  (red/amber = outside training distribution)",
                  color="#e2e8f0",pad=10)
    ax5.legend(fontsize=8,framealpha=0); ax5.grid(axis="x",alpha=0.3)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig5); plt.close()

# ── TAB 7 — Model Comparison ──────────────────────────────
with tab7:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#7c6af7;"></div>
        <div style="font-size:1rem;font-weight:600;">Model Comparison</div>
    </div>""", unsafe_allow_html=True)

    model_colors=[MUTED,MUTED,CYAN,GREEN,AMBER,VIOLET][:len(base_df)]
    c1,c2=st.columns(2)
    with c1:
        st.markdown("##### ROC-AUC by model")
        fig6,ax6=plt.subplots(figsize=(6,3.5))
        bars=ax6.barh(base_df["model"],base_df["roc_auc"],
                      color=model_colors,alpha=0.85,height=0.55,edgecolor="none")
        for bar,val in zip(bars,base_df["roc_auc"]):
            ax6.text(val+0.002,bar.get_y()+bar.get_height()/2,
                     f"{val:.4f}",va="center",fontsize=8,color="#e2e8f0")
        ax6.set_xlim(0.4,1.05); ax6.set_xlabel("ROC-AUC")
        ax6.grid(axis="x",alpha=0.3)
        ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig6); plt.close()
    with c2:
        st.markdown("##### Composite R² by model")
        fig7,ax7=plt.subplots(figsize=(6,3.5))
        bars2=ax7.barh(base_df["model"],base_df["composite_r2"],
                       color=model_colors,alpha=0.85,height=0.55,edgecolor="none")
        for bar,val in zip(bars2,base_df["composite_r2"]):
            ax7.text(val+0.005,bar.get_y()+bar.get_height()/2,
                     f"{val:.4f}",va="center",fontsize=8,color="#e2e8f0")
        ax7.set_xlim(-0.6,1.1); ax7.axvline(0,color=MUTED,lw=0.8,ls="--",alpha=0.5)
        ax7.set_xlabel("Composite R²"); ax7.grid(axis="x",alpha=0.3)
        ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig7); plt.close()

    st.markdown("##### Weight sensitivity — 500 Monte Carlo configurations")
    spread=sens_df["composite_mean"].max()-sens_df["composite_mean"].min()
    c1,c2,c3=st.columns(3)
    c1.markdown(f"""<div class="card"><div class="card-header">Score spread</div>
        <div class="card-value" style="color:#f0a020;">{spread:.4f}</div>
        <div class="card-sub">Moderate sensitivity</div></div>""",unsafe_allow_html=True)
    c2.markdown(f"""<div class="card"><div class="card-header">Default R²</div>
        <div class="card-value" style="color:#00d68a;">0.946</div>
        <div class="card-sub">Composite score quality</div></div>""",unsafe_allow_html=True)
    c3.markdown(f"""<div class="card"><div class="card-header">MC iterations</div>
        <div class="card-value" style="color:#00c8f0;">500</div>
        <div class="card-sub">Dirichlet-sampled</div></div>""",unsafe_allow_html=True)

    fig8,ax8=plt.subplots(figsize=(10,3))
    ax8.hist(sens_df["composite_mean"],bins=40,color=VIOLET,alpha=0.8,edgecolor="none")
    ax8.axvline(sens_df["composite_mean"].mean(),color=CYAN,lw=1.5,ls="--",
                label="Mean across configs")
    ax8.set_xlabel("Mean composite score under random weights")
    ax8.set_ylabel("Count"); ax8.legend(fontsize=8,framealpha=0)
    ax8.grid(axis="y",alpha=0.3)
    ax8.spines["top"].set_visible(False); ax8.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig8); plt.close()
    st.dataframe(base_df,use_container_width=True)

# ── TAB 8 — About & Scope ────────────────────────────────
with tab8:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;
    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2535;">
        <div style="width:8px;height:8px;border-radius:50%;background:#f0a020;"></div>
        <div style="font-size:1rem;font-weight:600;">About FormulAI — Scope & Limitations</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="disclosure-box">
        <div class="disclosure-title">Proof-of-concept disclosure — read before use</div>
        <div class="disclosure-text">
        FormulAI is trained entirely on <strong>synthetic, rule-based data</strong>.
        The model predicts outcomes generated by a biophysical simulation — it has not
        been validated against real wet-lab experimental measurements.<br><br>
        The reported ROC-AUC of 0.988 reflects performance within the simulation.
        Published models on real protein formulation data achieve ROC-AUC 0.78–0.88
        (Gentiluomo et al. 2022, Schmit et al. 2021). A performance drop to this range
        is expected when retrained on real data — and would confirm the model is
        learning biology rather than a simulation of it.
        </div>
    </div>""",unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown("#### Appropriate uses")
        st.markdown("""
- Internal research and hypothesis generation
- Prioritising formulations for experimental testing
- Grant proposals and investor demonstrations
- Reducing experimental search space before lab work
        """)
        st.markdown("#### Not appropriate for")
        st.markdown("""
- Replacing experimental stability studies (ICH Q1A)
- Regulatory submissions
- Clinical-stage molecule formulation decisions
- Any context where wrong predictions have safety consequences
        """)
    with col2:
        st.markdown("#### Known limitations")
        st.markdown("""
- Synthetic labels — model learns simulation, not biology
- Oxidation R²=0.594 — weakest target, needs AlphaFold SASA
- 40 proteins, 0.28% formulation space coverage
- Validated on small soluble human proteins only
- Not tested on mRNA-LNP, AAV, bispecifics, or ADCs
- Tree models interpolate poorly outside 4°C/25°C/40°C
- A2RU14 and A6BM72 flagged as out-of-distribution
        """)
        st.markdown("#### Negative findings (documented)")
        st.markdown("""
- SASA proxy features degraded oxidation R² from 0.624 → 0.034
- Multicollinearity with existing sequence features
- AlphaFold 3D coordinates required for real improvement
        """)

    st.markdown("---")
    st.markdown("""<div style="font-size:0.75rem;color:#6b748a;line-height:1.8;">
    <strong style="color:#9ca3af;">Architecture:</strong>
    Stacked ensemble — Ridge + XGBoost (monotone) + LightGBM → Ridge meta-learner<br>
    <strong style="color:#9ca3af;">Validation:</strong>
    Protein-level LOO — imputer and scaler fit inside each fold<br>
    <strong style="color:#9ca3af;">Uncertainty:</strong>
    Mean absolute disagreement across 3 base model pairs<br>
    <strong style="color:#9ca3af;">OOD detection:</strong>
    Mahalanobis distance — 95th percentile threshold<br>
    <strong style="color:#9ca3af;">Bias correction:</strong>
    LOO residual-estimated per-target offset applied to all predictions<br>
    <strong style="color:#9ca3af;">Data:</strong>
    UniProt sequences · PubChem excipient properties · Synthetic stability simulation<br>
    <strong style="color:#9ca3af;">Version:</strong> FormulAI v0.3 — pre-publication prototype
    </div>""",unsafe_allow_html=True)
