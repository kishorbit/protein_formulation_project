import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

st.set_page_config(
    page_title="FormulAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #05070d;
    color: #e2e8f0;
}
.stApp { background-color: #05070d; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0d1120 100%);
    border-right: 1px solid #1e2535;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.card-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b748a;
    margin-bottom: 0.4rem;
}
.card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}
.card-sub {
    font-size: 0.75rem;
    color: #6b748a;
    margin-top: 0.3rem;
}

/* Confidence badges */
.badge-high {
    background: rgba(0,214,138,0.15);
    color: #00d68a;
    border: 1px solid rgba(0,214,138,0.3);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-medium {
    background: rgba(240,160,32,0.15);
    color: #f0a020;
    border: 1px solid rgba(240,160,32,0.3);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-low {
    background: rgba(240,80,112,0.15);
    color: #f05070;
    border: 1px solid rgba(240,80,112,0.3);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1e2535;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00c8f0;
    box-shadow: 0 0 8px #00c8f0;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* Logo */
.logo-block {
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}
.logo-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 0.72rem;
    color: #6b748a;
    margin-top: 2px;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

/* Table styling */
.dataframe { font-size: 0.8rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    color: #6b748a;
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 8px;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,200,240,0.1) !important;
    color: #00c8f0 !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Dark matplotlib theme ─────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#05070d",
    "axes.facecolor":    "#0a0e1a",
    "axes.edgecolor":    "#1e2535",
    "axes.labelcolor":   "#6b748a",
    "xtick.color":       "#6b748a",
    "ytick.color":       "#6b748a",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2535",
    "grid.linewidth":    0.5,
    "font.family":       "sans-serif",
    "font.size":         9,
})

CYAN   = "#00c8f0"
GREEN  = "#00d68a"
AMBER  = "#f0a020"
VIOLET = "#7c6af7"
ROSE   = "#f05070"
MUTED  = "#6b748a"
TARGET_COLORS = [ROSE, AMBER, VIOLET, GREEN, CYAN]

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    recs    = pd.read_csv("outputs/reports/excipient_recommendations.csv")
    pareto  = pd.read_csv("outputs/reports/pareto_recommendations.csv")
    shap_df = pd.read_csv("outputs/reports/shap_importance.csv")
    merged  = pd.read_csv("data/processed/dataset_merged.csv")
    prot    = pd.read_csv("data/processed/protein_features.csv")
    val_df  = pd.read_csv("outputs/reports/held_out_validation.csv")
    base_df = pd.read_csv("outputs/reports/baseline_comparison.csv")
    return recs, pareto, shap_df, merged, prot, val_df, base_df

recs, pareto, shap_df, merged, prot_df, val_df, base_df = load_data()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-title">🧬 FormulAI</div>
        <div class="logo-sub">Protein Formulation Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    proteins   = sorted(merged["protein_id"].unique().tolist())
    sel_prot   = st.selectbox("Select Protein", proteins)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#6b748a;font-weight:600;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
    Pipeline Stats
    </div>
    """, unsafe_allow_html=True)

    valid_auc  = val_df["roc_auc"].dropna()
    n_stable   = val_df["roc_auc"].isna().sum()
    mean_auc   = valid_auc.mean()
    min_auc    = valid_auc.min()

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:0.6rem;">
        <div style="display:flex;justify-content:space-between;
        font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Proteins</span>
            <span style="color:#e2e8f0;font-weight:600;">{len(proteins)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;
        font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Mean ROC-AUC</span>
            <span style="color:#00d68a;font-weight:600;">{mean_auc:.3f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;
        font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Min ROC-AUC</span>
            <span style="color:#00c8f0;font-weight:600;">{min_auc:.3f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;
        font-size:0.78rem;padding:6px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#6b748a;">Inherently Stable</span>
            <span style="color:#f0a020;font-weight:600;">{n_stable}/40</span>
        </div>
        <div style="display:flex;justify-content:space-between;
        font-size:0.78rem;padding:6px 0;">
            <span style="color:#6b748a;">Architecture</span>
            <span style="color:#7c6af7;font-weight:600;">Stacked</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("UniProt · PubChem · FDA DailyMed · PubMed")

# ── Main tabs ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Recommendations",
    "Pareto Front",
    "SHAP Explainability",
    "Protein Profile",
    "Validation",
    "Model Comparison",
])

# ── TAB 1 — Recommendations ──────────────────────────────
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <div class="section-title">Excipient Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    sub = recs[recs["protein_id"] == sel_prot].copy()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        best = sub["pred_composite_score"].min()
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Best Composite Score</div>
            <div class="card-value" style="color:#00d68a;">{best:.3f}</div>
            <div class="card-sub">Lower = more stable</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        n_high = (sub.get("confidence","") == "High").sum() \
                 if "confidence" in sub.columns else 0
        st.markdown(f"""
        <div class="card">
            <div class="card-header">High Confidence</div>
            <div class="card-value" style="color:#00c8f0;">{n_high}</div>
            <div class="card-sub">All 3 models agree</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        n_stable_recs = sub.get("pred_stable", pd.Series([0])).sum() \
                        if "pred_stable" in sub.columns else 0
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Stable Formulations</div>
            <div class="card-value" style="color:#7c6af7;">{int(n_stable_recs)}</div>
            <div class="card-sub">of {len(sub)} total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        best_agg = sub["pred_aggregation"].min() \
                   if "pred_aggregation" in sub.columns else 0
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Best Aggregation</div>
            <div class="card-value" style="color:#f0a020;">{best_agg:.3f}</div>
            <div class="card-sub">Predicted score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("##### Top 20 Recommendations")

    show_cols = [c for c in [
        "buffer","sugar","surfactant","amino_acid","ph","temperature_c",
        "pred_composite_score","pred_aggregation","pred_oxidation",
        "pred_deamidation","pred_potency","confidence"
    ] if c in sub.columns]

    display = sub[show_cols].head(20).reset_index(drop=True)

    # Color confidence column
    def style_confidence(val):
        if val == "High":   return "color: #00d68a; font-weight: 600"
        if val == "Medium": return "color: #f0a020; font-weight: 600"
        if val == "Low":    return "color: #f05070; font-weight: 600"
        return ""

    if "confidence" in display.columns:
        styled = display.style.applymap(
            style_confidence, subset=["confidence"]
        ).format({c: "{:.3f}" for c in display.select_dtypes("float").columns})
        st.dataframe(styled, use_container_width=True, height=420)
    else:
        st.dataframe(display, use_container_width=True, height=420)

# ── TAB 2 — Pareto Front ─────────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#7c6af7;box-shadow:0 0 8px #7c6af7;"></div>
        <div class="section-title">Pareto-Optimal Formulations</div>
    </div>
    """, unsafe_allow_html=True)

    psub = pareto[pareto["protein_id"] == sel_prot].copy()

    if len(psub) == 0:
        st.info("No Pareto-optimal formulations found for this protein.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Pareto Formulations</div>
                <div class="card-value" style="color:#7c6af7;">{len(psub)}</div>
                <div class="card-sub">Non-dominated solutions</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            best_comp = psub["composite_stability_score"].min()
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Best Composite</div>
                <div class="card-value" style="color:#00d68a;">{best_comp:.4f}</div>
                <div class="card-sub">Lowest degradation risk</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            n_high_p = (psub.get("confidence","") == "High").sum()
            st.markdown(f"""
            <div class="card">
                <div class="card-header">High Confidence</div>
                <div class="card-value" style="color:#00c8f0;">{n_high_p}</div>
                <div class="card-sub">Pareto + all models agree</div>
            </div>""", unsafe_allow_html=True)

        # Radar-style bar chart for top 5
        top5 = psub.head(5)
        if len(top5) > 0:
            st.markdown("##### Score breakdown — top 5 Pareto formulations")
            labels  = [f"{r['buffer'][:8]}/{r['sugar'][:6]}"
                       for _, r in top5.iterrows()]
            metrics = [
                ("stability_score",  "Stability",      CYAN),
                ("low_aggregation",  "Low Aggregation",GREEN),
                ("low_oxidation",    "Low Oxidation",  AMBER),
                ("low_deamidation",  "Low Deamidation",VIOLET),
                ("high_potency",     "Potency",        ROSE),
            ]
            metrics = [(m,l,c) for m,l,c in metrics if m in top5.columns]

            fig, axes = plt.subplots(1, len(metrics),
                                     figsize=(14, 3.5))
            if len(metrics) == 1: axes = [axes]
            for ax, (metric, label, color) in zip(axes, metrics):
                vals = top5[metric].values
                bars = ax.bar(range(len(top5)), vals,
                              color=color, alpha=0.8, width=0.6,
                              edgecolor="none")
                ax.set_title(label, fontsize=9, color="#e2e8f0", pad=8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=35,
                                   ha="right", fontsize=7)
                ax.set_ylim(0, 1.1)
                ax.axhline(1.0, color=MUTED, lw=0.5, ls="--", alpha=0.5)
                ax.grid(axis="y", alpha=0.3)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            val+0.03, f"{val:.2f}",
                            ha="center", fontsize=7, color="#e2e8f0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        show_p = [c for c in [
            "buffer","sugar","surfactant","amino_acid","ph",
            "composite_stability_score","stability_score",
            "low_aggregation","low_oxidation","high_potency","confidence"
        ] if c in psub.columns]
        st.dataframe(psub[show_p].reset_index(drop=True),
                     use_container_width=True, height=300)

# ── TAB 3 — SHAP ─────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#f0a020;box-shadow:0 0 8px #f0a020;"></div>
        <div class="section-title">Feature Importance (SHAP)</div>
    </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("Show top N features", 5, 30, 15)
    top   = shap_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1))
    colors  = [CYAN if i < 5 else GREEN if i < 10 else MUTED
               for i in range(len(top))]
    bars = ax.barh(top["feature"][::-1],
                   top["shap_importance"][::-1],
                   color=colors[::-1], alpha=0.85,
                   height=0.65, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|", color=MUTED)
    ax.set_title("What drives aggregation predictions",
                 color="#e2e8f0", pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class="card">
        <div class="card-header">Cyan</div>
        <div style="color:#00c8f0;font-size:0.82rem;">Top 5 drivers</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="card">
        <div class="card-header">Green</div>
        <div style="color:#00d68a;font-size:0.82rem;">Features 6–10</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="card">
        <div class="card-header">Gray</div>
        <div style="color:#6b748a;font-size:0.82rem;">Lower importance</div>
    </div>""", unsafe_allow_html=True)

# ── TAB 4 — Protein Profile ───────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#00d68a;box-shadow:0 0 8px #00d68a;"></div>
        <div class="section-title">Protein Physicochemical Profile</div>
    </div>
    """, unsafe_allow_html=True)

    matches = prot_df[prot_df["protein_id"] == sel_prot]
    if len(matches) == 0:
        st.warning("No profile found for this protein.")
    else:
        prow = matches.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, key, color in [
            (c1, "Isoelectric Point", "isoelectric_point", CYAN),
            (c2, "GRAVY Score",       "gravy_score",       GREEN),
            (c3, "Instability Index", "instability_index", AMBER),
            (c4, "Agg Mean Score",    "agg_mean",          ROSE),
        ]:
            val = prow.get(key, "N/A")
            val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
            col.markdown(f"""
            <div class="card">
                <div class="card-header">{label}</div>
                <div class="card-value" style="color:{color};font-size:1.6rem;">
                    {val_str}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("##### Aggregation risk profile")
        agg_data = {
            "Mean agg score":    prow.get("agg_mean",          np.nan),
            "Max agg score":     prow.get("agg_max",           np.nan),
            "Hotspot count":     prow.get("agg_hotspots",      np.nan),
            "Hotspot fraction":  prow.get("agg_hotspot_frac",  np.nan),
        }
        st.dataframe(
            pd.DataFrame(agg_data, index=["value"]).T,
            use_container_width=True)

        st.markdown("##### Degradation-risk residue content")
        risk = {}
        for aa, label in [
            ("pct_asn","Asn (deamidation)"),
            ("pct_met","Met (oxidation)"),
            ("pct_cys","Cys (disulfide)"),
            ("pct_trp","Trp (oxidation)"),
            ("pct_his","His (pH-sensitive)"),
        ]:
            if aa in prow.index and not pd.isna(prow[aa]):
                risk[label] = prow[aa]

        if risk:
            fig2, ax2 = plt.subplots(figsize=(7, 2.8))
            ax2.bar(risk.keys(), risk.values(),
                    color=TARGET_COLORS[:len(risk)],
                    alpha=0.85, width=0.5, edgecolor="none")
            ax2.set_ylabel("% of sequence", color=MUTED)
            ax2.set_title("Residues linked to chemical degradation",
                          color="#e2e8f0", pad=10)
            ax2.grid(axis="y", alpha=0.3)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ── TAB 5 — Validation ────────────────────────────────────
with tab5:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#f05070;box-shadow:0 0 8px #f05070;"></div>
        <div class="section-title">Held-Out Validation — Stacked Ensemble</div>
    </div>
    """, unsafe_allow_html=True)

    valid   = val_df[val_df["roc_auc"].notna()]
    n_nan   = val_df["roc_auc"].isna().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Mean ROC-AUC</div>
            <div class="card-value" style="color:#00d68a;">
                {valid['roc_auc'].mean():.3f}
            </div>
            <div class="card-sub">{len(valid)} proteins evaluated</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Min ROC-AUC</div>
            <div class="card-value" style="color:#00c8f0;">
                {valid['roc_auc'].min():.3f}
            </div>
            <div class="card-sub">Worst-case protein</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Mean Composite R²</div>
            <div class="card-value" style="color:#7c6af7;">
                {val_df['composite_r2'].mean():.3f}
            </div>
            <div class="card-sub">All 40 proteins</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Inherently Stable</div>
            <div class="card-value" style="color:#f0a020;">{n_nan}</div>
            <div class="card-sub">Single-class proteins</div>
        </div>""", unsafe_allow_html=True)

    # AUC bar chart
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    auc_vals = val_df["roc_auc"].fillna(0).values
    colors_bar = []
    for v, pid in zip(auc_vals, val_df["held_out_protein"]):
        if val_df[val_df["held_out_protein"]==pid]["roc_auc"].isna().values[0]:
            colors_bar.append(AMBER)
        elif v >= 0.97:
            colors_bar.append(GREEN)
        elif v >= 0.90:
            colors_bar.append(CYAN)
        else:
            colors_bar.append(ROSE)

    ax3.bar(range(len(val_df)), auc_vals,
            color=colors_bar, alpha=0.85,
            width=0.7, edgecolor="none")
    ax3.axhline(0.97, color=GREEN, ls="--", lw=1,
                alpha=0.6, label="0.97")
    ax3.axhline(0.90, color=CYAN,  ls="--", lw=1,
                alpha=0.6, label="0.90")
    ax3.set_xticks(range(len(val_df)))
    ax3.set_xticklabels(val_df["held_out_protein"],
                        rotation=90, fontsize=6.5)
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title("Per-protein LOO ROC-AUC  "
                  "(amber = inherently stable, no AUC computable)",
                  color="#e2e8f0", pad=10)
    ax3.legend(fontsize=8, framealpha=0)
    ax3.set_ylim(0, 1.08)
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("##### Full validation table")
    st.dataframe(val_df, use_container_width=True, height=300)

# ── TAB 6 — Model Comparison ──────────────────────────────
with tab6:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#7c6af7;box-shadow:0 0 8px #7c6af7;"></div>
        <div class="section-title">Model Comparison — Baseline vs Stacked</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### ROC-AUC by model")
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        model_colors = [MUTED, MUTED, CYAN, GREEN, AMBER, VIOLET]
        model_colors = model_colors[:len(base_df)]
        bars = ax4.barh(base_df["model"], base_df["roc_auc"],
                        color=model_colors, alpha=0.85,
                        height=0.55, edgecolor="none")
        ax4.axvline(0.5, color=MUTED, lw=0.8, ls="--", alpha=0.5)
        for bar, val in zip(bars, base_df["roc_auc"]):
            ax4.text(val+0.002, bar.get_y()+bar.get_height()/2,
                     f"{val:.4f}", va="center",
                     fontsize=8, color="#e2e8f0")
        ax4.set_xlim(0.4, 1.05)
        ax4.set_xlabel("ROC-AUC")
        ax4.grid(axis="x", alpha=0.3)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with c2:
        st.markdown("##### Composite R² by model")
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        bars2 = ax5.barh(base_df["model"], base_df["composite_r2"],
                         color=model_colors, alpha=0.85,
                         height=0.55, edgecolor="none")
        for bar, val in zip(bars2, base_df["composite_r2"]):
            ax5.text(val+0.005, bar.get_y()+bar.get_height()/2,
                     f"{val:.4f}", va="center",
                     fontsize=8, color="#e2e8f0")
        ax5.set_xlim(-0.6, 1.1)
        ax5.axvline(0, color=MUTED, lw=0.8, ls="--", alpha=0.5)
        ax5.set_xlabel("Composite R²")
        ax5.grid(axis="x", alpha=0.3)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    st.markdown("##### Full comparison table")
    st.dataframe(base_df, use_container_width=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <div class="card-header">Architecture</div>
        <div style="font-size:0.82rem;color:#e2e8f0;line-height:1.8;">
            <span style="color:#00c8f0;">Layer 1</span> —
            Ridge + XGBoost (monotone) + LightGBM trained independently<br>
            <span style="color:#7c6af7;">Layer 2</span> —
            Ridge meta-learner on 15 base model outputs<br>
            <span style="color:#00d68a;">Uncertainty</span> —
            Mean absolute disagreement across model pairs<br>
            <span style="color:#f0a020;">Validation</span> —
            Protein-level leave-one-out (impute + scale inside fold)
        </div>
    </div>
    """, unsafe_allow_html=True)
