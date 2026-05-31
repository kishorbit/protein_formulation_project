import streamlit as st
from fpdf import FPDF
import io
from datetime import datetime
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict_api import (
    predict_single, ALLOWED, _pf,
    _cls_model, _reg_model, _feature_cols, _build_vector
)

st.set_page_config(page_title="PredStabio", page_icon="⬡", layout="centered")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #e2e4ed !important;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: #07080f !important; }

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}

.stApp::after {
    content: '';
    position: fixed;
    top: -200px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 400px;
    background: radial-gradient(ellipse, rgba(0,122,255,0.08) 0%, transparent 70%);
    z-index: 0;
    pointer-events: none;
}

.stApp > * { position: relative; z-index: 1; }

section[data-testid="stSidebar"] { display: none !important; }
.stApp > header { background: transparent !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

.wordmark {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 56px 0 40px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 32px;
    position: relative;
}
.wordmark::before {
    content: '';
    position: absolute;
    top: 20px; left: 50%;
    transform: translateX(-50%);
    width: 320px; height: 160px;
    background: radial-gradient(ellipse, rgba(0,122,255,0.10) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.wordmark-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, rgba(0,122,255,0.20), rgba(0,60,140,0.10));
    border: 1px solid rgba(0,122,255,0.30);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px;
    box-shadow: 0 0 24px rgba(0,122,255,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative; z-index: 1;
}
.wordmark-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #ffffff;
    position: relative; z-index: 1;
}
.wordmark-name span { color: #007AFF; }
.wordmark-sub {
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.22);
    margin-top: 10px;
    position: relative; z-index: 1;
}
.wordmark-pills {
    display: flex; gap: 8px; margin-top: 16px; position: relative; z-index: 1;
}
.wordmark-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(0,122,255,0.7);
    border: 1px solid rgba(0,122,255,0.20);
    border-radius: 20px;
    padding: 3px 10px;
    background: rgba(0,122,255,0.06);
}
.wordmark-rule {
    width: 32px; height: 1px;
    background: linear-gradient(90deg, transparent, #007AFF, transparent);
    margin-top: 16px; opacity: 0.6;
    position: relative; z-index: 1;
}

.step-rail {
    display: flex;
    align-items: center;
    margin-bottom: 32px;
    padding: 0 4px;
}
.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}
.step-circle {
    width: 28px; height: 28px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.12);
    background: #0d0e18;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.25);
    position: relative;
    z-index: 2;
    transition: all 0.3s;
}
.step-circle.active {
    border-color: #007AFF;
    background: rgba(0,122,255,0.15);
    color: #007AFF;
    box-shadow: 0 0 0 3px rgba(0,122,255,0.12);
}
.step-circle.done {
    border-color: rgba(0,122,255,0.4);
    background: rgba(0,122,255,0.08);
    color: rgba(0,122,255,0.7);
}
.step-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: rgba(255,255,255,0.20);
    margin-top: 8px;
    text-align: center;
    text-transform: uppercase;
}
.step-label.active { color: #007AFF; }
.step-label.done   { color: rgba(0,122,255,0.55); }
.step-connector {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin-top: -18px;
    position: relative;
    z-index: 1;
}
.step-connector.done { background: rgba(0,122,255,0.3); }

.panel {
    background: #0d0e18;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 24px 28px;
    margin-bottom: 12px;
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
.panel-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px; height: 40px;
    background: linear-gradient(135deg, rgba(0,122,255,0.18) 0%, rgba(0,80,180,0.08) 100%);
    border: 1px solid rgba(0,122,255,0.30);
    border-radius: 10px;
    font-size: 15px;
    box-shadow: 0 0 12px rgba(0,122,255,0.20), inset 0 1px 0 rgba(255,255,255,0.08);
    flex-shrink: 0;
}
.panel-badge svg {
    width: 20px; height: 20px;
    filter: drop-shadow(0 0 4px rgba(0,122,255,0.6));
}
.panel-title {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.01em;
}
.panel-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: rgba(0,122,255,0.60);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

.notice {
    background: rgba(0,122,255,0.05);
    border-left: 2px solid rgba(0,122,255,0.35);
    border-radius: 0 3px 3px 0;
    padding: 12px 16px;
    font-size: 12.5px;
    color: rgba(180,200,255,0.65);
    line-height: 1.6;
    margin-bottom: 4px;
}
.notice b { color: rgba(0,122,255,0.85); font-weight: 500; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.metric-cell {
    background: #0a0b14;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 3px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
.metric-cell:hover { border-color: rgba(0,122,255,0.25); }
.metric-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
    margin-bottom: 10px;
}
.metric-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #e2e4ed;
    line-height: 1;
}

.card-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 10px;
    margin-bottom: 12px;
}
.info-card {
    background: #0a0b14;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 3px;
    padding: 16px 18px 18px;
}
.info-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(0,122,255,0.65);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.info-card-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 3px;
    background: rgba(0,122,255,0.6);
    border-radius: 50%;
}
.info-card-body {
    font-size: 12.5px;
    color: rgba(255,255,255,0.38);
    line-height: 1.65;
}

.tag {
    display: inline-block;
    background: rgba(0,122,255,0.08);
    border: 1px solid rgba(0,122,255,0.18);
    border-radius: 2px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: rgba(100,160,255,0.80);
    margin: 2px 2px;
}

.divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 16px 0;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectSlider"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    color: rgba(255,255,255,0.50) !important;
    text-transform: uppercase !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0a0b14 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 3px !important;
    color: #e2e4ed !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(0,122,255,0.35) !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(0,122,255,0.55) !important;
    box-shadow: 0 0 0 2px rgba(0,122,255,0.10) !important;
}
.stSlider > div > div > div > div { background: #007AFF !important; }
[data-testid="stThumbValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #007AFF !important;
    font-size: 12px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #007AFF !important;
    border: 2px solid #0d0e18 !important;
}
button[kind="primary"] {
    background: #007AFF !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    color: #ffffff !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}
button[kind="primary"]:hover { background: #0066dd !important; }
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    color: rgba(255,255,255,0.45) !important;
    transition: all 0.15s !important;
}
button[kind="secondary"]:hover {
    border-color: rgba(255,255,255,0.22) !important;
    color: rgba(255,255,255,0.65) !important;
}
.stSpinner > div { border-top-color: #007AFF !important; }
[data-testid="stCheckbox"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: rgba(255,255,255,0.45) !important;
    font-size: 12.5px !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 3px !important;
}
/* ═══════════════════════════════
   TABS — Glassmorphism + Curved
═══════════════════════════════ */
div[data-testid="stTabs"] {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 16px;
    padding: 12px 12px 20px 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15);
}
div[data-testid="stTabs"] > div:first-child {
    gap: 8px;
    margin-bottom: 16px;
}
button[data-baseweb="tab"] {
    border-radius: 10px !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    color: rgba(255,255,255,0.70) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover {
    background: rgba(0,122,255,0.12) !important;
    border-color: rgba(0,122,255,0.30) !important;
    color: rgba(255,255,255,0.75) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(0,122,255,0.28) !important;
    border-color: rgba(0,122,255,0.70) !important;
    color: #4da6ff !important;
    box-shadow: 0 0 20px rgba(0,122,255,0.30), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}



/* ═══════════════════════════════
   GLASSMORPHISM PANELS
═══════════════════════════════ */
.panel {
    background: rgba(13,14,24,0.6) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
.metric-cell {
    background: rgba(10,11,20,0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.info-card {
    background: rgba(10,11,20,0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.tag {
    border-radius: 8px !important;
    border: 1px solid rgba(0,122,255,0.30) !important;
    background: rgba(0,122,255,0.10) !important;
    backdrop-filter: blur(4px) !important;
}

[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

div[data-testid="stTabsContent"] {
    background: rgba(13, 14, 24, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 0 16px 16px 16px !important;
    padding: 24px 28px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
</style>""", unsafe_allow_html=True)

# ── Wordmark ─────────────────────────────────────────────
st.markdown("""
<div class="wordmark">
  <div class="wordmark-icon">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="4"   r="2.2" fill="#007AFF" opacity="0.95"/>
      <circle cx="20" cy="8.5" r="2.2" fill="#007AFF" opacity="0.75"/>
      <circle cx="20" cy="15.5" r="2.2" fill="#34d399" opacity="0.75"/>
      <circle cx="12" cy="20" r="2.2" fill="#34d399" opacity="0.95"/>
      <circle cx="4"  cy="15.5" r="2.2" fill="#007AFF" opacity="0.75"/>
      <circle cx="4"  cy="8.5"  r="2.2" fill="#007AFF" opacity="0.75"/>
      <line x1="12" y1="4"   x2="20" y2="8.5"  stroke="rgba(0,122,255,0.35)" stroke-width="1"/>
      <line x1="20" y1="8.5" x2="20" y2="15.5" stroke="rgba(0,122,255,0.35)" stroke-width="1"/>
      <line x1="20" y1="15.5" x2="12" y2="20"  stroke="rgba(52,211,153,0.35)" stroke-width="1"/>
      <line x1="12" y1="20"  x2="4"  y2="15.5" stroke="rgba(52,211,153,0.35)" stroke-width="1"/>
      <line x1="4"  y1="15.5" x2="4" y2="8.5"  stroke="rgba(0,122,255,0.35)" stroke-width="1"/>
      <line x1="4"  y1="8.5"  x2="12" y2="4"   stroke="rgba(0,122,255,0.35)" stroke-width="1"/>
    </svg>
  </div>
  <div class="wordmark-name">PredStabio<span>™</span></div>
  <div class="wordmark-sub">AI-Powered Protein Stability Prediction Platform</div>
  <div class="wordmark-pills">
    <span class="wordmark-pill">XGBoost</span>
    <span class="wordmark-pill">SHAP</span>
    <span class="wordmark-pill">4,320 Formulations</span>
  </div>
  <div class="wordmark-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────
defaults = {
    "step": 1, "result": None,
    "buffer":            ALLOWED["buffer"][0],
    "sugar":             ALLOWED["sugar"][0],
    "surfactant":        ALLOWED["surfactant"][0],
    "amino_acid":        ALLOWED["amino_acid"][0],
    "salt":              ALLOWED["salt"][0],
    "ph":                ALLOWED["ph"][len(ALLOWED["ph"])//2],
    "temperature_c":     ALLOWED["temperature_c"][1],
    "protein_conc_mgmL": ALLOWED["protein_conc_mgmL"][1],
    "protein_id":        _pf["protein_id"].iloc[0],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def goto(n): st.session_state.step = n

def generate_pdf_report(r):
    pdf = FPDF()
    pdf.add_page()
    grade_colors = {"A":(52,211,153),"B":(163,230,53),"C":(251,191,36),"D":(249,115,22),"F":(248,113,113)}
    gc = grade_colors.get(r["stability_grade"], (226,228,237))

    # Header bar
    pdf.set_fill_color(13,14,24)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "PredStabio - Stability Assessment Report", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(150,150,180)
    pdf.set_x(10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  AI-Powered Protein Formulation Platform", ln=True)

    pdf.set_y(35)

    # Protein info
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0,122,255)
    pdf.cell(0, 6, "PROTEIN", ln=True)
    pdf.set_draw_color(0,122,255)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40,40,60)
    pdf.cell(50, 7, "Protein ID:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, str(r["protein_id"]), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 7, "Protein Type:")
    pdf.cell(0, 7, str(r["protein_type"]), ln=True)
    pdf.ln(4)

    # Grade box
    pdf.set_fill_color(*gc)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "STABILITY ASSESSMENT", ln=True)
    pdf.set_draw_color(*gc)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    # Grade large
    pdf.set_fill_color(*gc)
    pdf.set_text_color(*gc)
    pdf.set_font("Helvetica", "B", 48)
    pdf.set_x(10)
    pdf.cell(30, 20, r["stability_grade"], border=0)
    grade_labels = {"A":"Excellent","B":"Good","C":"Marginal","D":"Poor","F":"Fail"}
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40,40,60)
    pdf.set_xy(45, pdf.get_y()+2)
    pdf.cell(0, 7, f"Grade: {grade_labels.get(r['stability_grade'],'')}",ln=True)
    pdf.set_x(45)
    pdf.cell(0, 7, f"Composite Score: {r['pred_composite_score']:.4f}", ln=True)
    pdf.set_x(45)
    pdf.cell(0, 7, f"P(Stable): {r['pred_stable_proba']:.4f}", ln=True)
    pdf.ln(6)

    # Formulation
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0,122,255)
    pdf.cell(0, 6, "FORMULATION USED", ln=True)
    pdf.set_draw_color(0,122,255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    skip = {"protein_id","protein_type"}
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40,40,60)
    for k, v in r["formulation"].items():
        if v is not None and k not in skip:
            label = k.replace("_", " ").title()
            pdf.cell(70, 7, f"{label}:")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, str(v), ln=True)
            pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)

    # Risk flags
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0,122,255)
    pdf.cell(0, 6, "RISK FLAGS", ln=True)
    pdf.set_draw_color(0,122,255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    if r["risk_flags"]:
        pdf.set_text_color(220,80,80)
        for fl in r["risk_flags"]:
            pdf.cell(0, 7, f"  [!]  {fl}", ln=True)
    else:
        pdf.set_text_color(52,180,120)
        pdf.cell(0, 7, "  [OK]  No risk flags detected", ln=True)
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150,150,170)
    pdf.set_fill_color(245,245,250)
    pdf.rect(10, pdf.get_y(), 190, 16, "F")
    pdf.set_xy(13, pdf.get_y()+3)
    pdf.multi_cell(184, 5, "DISCLAIMER: This report is generated from synthetic training data. Wet-lab validation is required before making formulation decisions. For research use only.")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

# ── Step rail ────────────────────────────────────────────
step_labels = ["Protein", "Excipients", "Conditions", "Results"]
current = st.session_state.step

rail = '<div class="step-rail">'
for i, label in enumerate(step_labels, 1):
    circle_cls = "active" if i == current else ("done" if i < current else "")
    label_cls  = "active" if i == current else ("done" if i < current else "")
    marker = "✓" if i < current else str(i)
    rail += f'''<div class="step-node">
        <div class="step-circle {circle_cls}">{marker}</div>
        <div class="step-label {label_cls}">{label}</div>
    </div>'''
    if i < 4:
        conn_cls = "done" if i < current else ""
        rail += f'<div class="step-connector {conn_cls}"></div>'
rail += '</div>'
st.markdown(rail, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# STEP 1 — Protein
# ════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 3C9 6 7 9 8 12C9 15 12 16 12 16" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/><path d="M12 3C15 6 17 9 16 12C15 15 12 16 12 16" stroke="#34d399" stroke-width="1.5" stroke-linecap="round"/><path d="M12 16C9 18 8 20 9 21" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/><path d="M12 16C15 18 16 20 15 21" stroke="#34d399" stroke-width="1.5" stroke-linecap="round"/><line x1="8.5" y1="7" x2="15.5" y2="7" stroke="rgba(0,122,255,0.4)" stroke-width="1"/><line x1="7.5" y1="12" x2="16.5" y2="12" stroke="rgba(52,211,153,0.4)" stroke-width="1"/></svg></div>
        <div>
          <div class="panel-title">Protein Parameters</div>
          <div class="panel-step">Step 01 / 03</div>
        </div>
      </div>
      <div class="notice">
        <b>Note:</b> Select your protein by UniProt ID. Biophysical properties are
        retrieved from our curated database. Aggregation propensity computed via
        Aggrescan3D / CamSol.
      </div>
    </div>
    """, unsafe_allow_html=True)

    protein_id = st.selectbox(
        "UNIPROT ID",
        options=_pf["protein_id"].tolist(),
        format_func=lambda x: f"{x}  —  {_pf[_pf['protein_id']==x]['query_label'].values[0]}",
        key="protein_id"
    )

    _match = _pf[_pf["protein_id"] == protein_id]
    prow = _match.iloc[0] if len(_match) > 0 else _pf.iloc[0]

    st.markdown(f"""
    <div class="panel" style="margin-top:8px">
      <div class="metric-grid">
        <div class="metric-cell">
          <div class="metric-key">Isoelectric Point (pI)</div>
          <div class="metric-num" style="color:#60a5fa">{prow['isoelectric_point']:.2f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Instability Index</div>
          <div class="metric-num" style="color:{'#fbbf24' if prow['instability_index']>40 else '#34d399'}">{prow['instability_index']:.1f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">GRAVY Score</div>
          <div class="metric-num" style="color:{'#f87171' if prow['gravy_score']>0 else '#34d399'}">{prow['gravy_score']:.3f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Met Exposed Frac.</div>
          <div class="metric-num">{prow['met_exposed_fraction']:.3f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Agg. Hotspot Frac.</div>
          <div class="metric-num">{prow['agg_hotspot_frac']:.3f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Ox. Risk Composite</div>
          <div class="metric-num">{prow['ox_risk_composite']:.3f}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-row">
      <div class="info-card">
        <div class="info-card-label">pI &amp; Aggregation</div>
        <div class="info-card-body">Proteins formulated within 2 pH units of their pI are highly prone to aggregation.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Instability Index</div>
        <div class="info-card-body">Guruprasad scale: &lt;33 stable, 33–40 borderline, &gt;40 unstable.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">GRAVY Score</div>
        <div class="info-card-body">Positive = hydrophobic (aggregation-prone). Negative = hydrophilic (stable).</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Continue →", type="primary", use_container_width=True):
        goto(2); st.rerun()


# ════════════════════════════════════════════════════════
# STEP 2 — Excipients
# ════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M9 3V13L5 19H19L15 13V3" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><line x1="9" y1="3" x2="15" y2="3" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/><circle cx="10" cy="16" r="1" fill="#34d399"/><circle cx="13" cy="18" r="0.8" fill="#34d399" opacity="0.7"/></svg></div>
        <div>
          <div class="panel-title">Excipient Composition</div>
          <div class="panel-step">Step 02 / 03</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("BUFFER SYSTEM",    ALLOWED["buffer"],     key="buffer")
        st.selectbox("SURFACTANT",        ALLOWED["surfactant"], key="surfactant")
        st.selectbox("SALT",              ALLOWED["salt"],       key="salt")
    with col2:
        st.selectbox("CRYOPROTECTANT / SUGAR", ALLOWED["sugar"],      key="sugar")
        st.selectbox("AMINO ACID STABILISER",  ALLOWED["amino_acid"], key="amino_acid")

    st.markdown("""
    <div class="card-row" style="margin-top:20px">
      <div class="info-card">
        <div class="info-card-label">Buffer Selection</div>
        <div class="info-card-body">Histidine preferred for pH 5–7 biologics. Citrate works at lower pH but can chelate metals.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Surfactants</div>
        <div class="info-card-body">PS80 provides stronger interface protection. PS20 is more oxidation-stable.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Methionine</div>
        <div class="info-card-body">Sacrificial oxidation substrate protecting Met residues from oxidative degradation.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", use_container_width=True):
            goto(1); st.rerun()
    with c2:
        if st.button("Continue →", type="primary", use_container_width=True):
            goto(3); st.rerun()


# ════════════════════════════════════════════════════════
# STEP 3 — Conditions
# ════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" stroke="#007AFF" stroke-width="1.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/></svg></div>
        <div>
          <div class="panel-title">Process Conditions</div>
          <div class="panel-step">Step 03 / 03</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ph   = st.select_slider("FORMULATION pH", options=ALLOWED["ph"], value=5.5, key="ph")
        temp = st.selectbox("STORAGE TEMPERATURE (°C)", ALLOWED["temperature_c"], index=1, key="temperature_c")
    with col2:
        conc = st.selectbox("PROTEIN CONCENTRATION (mg/mL)", ALLOWED["protein_conc_mgmL"], index=1, key="protein_conc_mgmL")

    pid    = st.session_state.get("protein_id") or _pf["protein_id"].iloc[0]
    _m3    = _pf[_pf["protein_id"] == pid]
    prow   = _m3.iloc[0] if len(_m3) > 0 else _pf.iloc[0]
    pi     = prow["isoelectric_point"]
    pi_dist = abs(ph - pi)
    pi_color = "#34d399" if pi_dist >= 2 else "#fbbf24" if pi_dist >= 1 else "#f87171"
    pi_status = "✓ Good electrostatic repulsion" if pi_dist >= 2 else "⚠ Marginal — aggregation risk" if pi_dist >= 1 else "✗ Too close to pI"

    buf = st.session_state.get('buffer') or '—'
    sug = st.session_state.get('sugar') or '—'
    sur = st.session_state.get('surfactant') or '—'
    aa  = st.session_state.get('amino_acid') or '—'
    sal = st.session_state.get('salt') or '—'

    st.markdown(f"""
    <div class="panel" style="margin-top:8px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(255,255,255,0.25);margin-bottom:12px">Formulation Summary</div>
      <div style="margin-bottom:14px">
        <span class="tag">{buf}</span>
        <span class="tag">{sug}</span>
        <span class="tag">{sur}</span>
        <span class="tag">{aa}</span>
        <span class="tag">{sal}</span>
        <span class="tag">pH {ph}</span>
        <span class="tag">{temp} °C</span>
        <span class="tag">{conc} mg/mL</span>
      </div>
      <div class="divider"></div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:12px">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(255,255,255,0.30)">|pH − pI|</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:500;color:{pi_color}">{pi_dist:.2f}</span>
        <span style="font-size:11.5px;color:rgba(255,255,255,0.30)">{pi_status}</span>
      </div>
    </div>
    <div class="card-row">
      <div class="info-card">
        <div class="info-card-label">pH &amp; Solubility</div>
        <div class="info-card-body">Formulating &gt;2 pH units from pI maximises electrostatic repulsion and colloidal stability.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Temperature</div>
        <div class="info-card-body">Each 10°C rise roughly doubles degradation rate (Arrhenius). Accelerated studies predict shelf life.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Concentration</div>
        <div class="info-card-body">Above 50 mg/mL requires special viscosity and aggregation attention.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", use_container_width=True):
            goto(2); st.rerun()
    with c2:
        if st.button("⬡ Run Assessment", type="primary", use_container_width=True):
            formulation = {
                "buffer":            st.session_state.buffer,
                "sugar":             st.session_state.sugar,
                "surfactant":        st.session_state.surfactant,
                "amino_acid":        st.session_state.amino_acid,
                "salt":              st.session_state.salt,
                "ph":                st.session_state.ph,
                "temperature_c":     st.session_state.temperature_c,
                "protein_conc_mgmL": st.session_state.protein_conc_mgmL,
            }
            with st.spinner("Running ML prediction..."):
                st.session_state.result = predict_single(pid, formulation)
            goto(4); st.rerun()


# ════════════════════════════════════════════════════════
# STEP 4 — Results
# ════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    r = st.session_state.result
    grade = r["stability_grade"]
    grade_colors = {"A":"#34d399","B":"#a3e635","C":"#fbbf24","D":"#f97316","F":"#f87171"}
    grade_labels = {"A":"Excellent","B":"Good","C":"Marginal","D":"Poor","F":"Fail"}
    color = grade_colors.get(grade, "#e2e4ed")
    score = r["pred_composite_score"]
    stable = r["pred_stable_proba"]
    bar_pct = int(score * 100)
    grade_label = grade_labels.get(grade, "")
    protein_id_disp = r["protein_id"]
    protein_type_disp = r["protein_type"]

    if r["risk_flags"]:
        flags_html = "".join(
            f'<span style="display:inline-block;background:rgba(255,59,48,0.08);border:1px solid rgba(255,59,48,0.20);border-radius:2px;padding:3px 10px;font-family:IBM Plex Mono,monospace;font-size:10.5px;color:rgba(255,130,120,0.80);margin:2px">{fl}</span>'
            for fl in r["risk_flags"]
        )
    else:
        flags_html = '<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:rgba(48,209,88,0.75)">✓ No risk flags detected</span>'

    skip = {"buf_conc_mM","sug_conc_mM","sur_conc_mM","aa_conc_mM","salt_conc_mM"}
    pills_html = "".join(
        f'<span style="display:inline-block;background:rgba(0,122,255,0.08);border:1px solid rgba(0,122,255,0.18);border-radius:2px;padding:3px 10px;font-family:IBM Plex Mono,monospace;font-size:11px;color:rgba(100,160,255,0.80);margin:2px 2px">{k}: {v}</span>'
        for k, v in r["formulation"].items()
        if v is not None and k not in skip
    )

    st.markdown(f"""<div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="14" width="3" height="7" rx="1" fill="#007AFF" opacity="0.5"/>
            <rect x="8" y="10" width="3" height="11" rx="1" fill="#007AFF" opacity="0.7"/>
            <rect x="13" y="6" width="3" height="15" rx="1" fill="#007AFF" opacity="0.9"/>
            <rect x="18" y="3" width="3" height="18" rx="1" fill="#34d399" opacity="0.85"/>
            <polyline points="4.5,13 9.5,9 14.5,5 19.5,2" stroke="#34d399" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.6"/>
          </svg></div>
        <div>
          <div class="panel-title">Assessment Results</div>
          <div class="panel-step">{protein_id_disp} · {protein_type_disp}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    gcol, mcol = st.columns([1, 2])
    with gcol:
        st.markdown(f"""<div class="panel" style="text-align:center;height:100%">
          <div style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(255,255,255,0.25);margin-bottom:12px;animation:fadeUp 0.5s ease both">Stability Grade</div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:80px;font-weight:600;line-height:1;letter-spacing:-0.04em;color:{color};animation:gradeReveal 0.7s cubic-bezier(0.34,1.56,0.64,1) both;animation-delay:0.15s;filter:drop-shadow(0 0 18px {color}55)">{grade}</div>
          <div style="font-size:12px;color:{color};opacity:0.65;margin-top:8px;font-weight:400;letter-spacing:0.10em;text-transform:uppercase;animation:fadeUp 0.5s ease both;animation-delay:0.35s">{grade_label}</div>
          <style>
            @keyframes gradeReveal {{
              from {{ opacity:0; transform:scale(0.6); filter:blur(8px); }}
              to   {{ opacity:1; transform:scale(1);   filter:blur(0);   }}
            }}
            @keyframes fadeUp {{
              from {{ opacity:0; transform:translateY(8px); }}
              to   {{ opacity:1; transform:translateY(0);   }}
            }}
          </style>
          <div style="margin-top:20px;padding:0 4px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.20)">Composite Score</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:10px;color:{color};opacity:0.8">{score:.4f}</span>
            </div>
            <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
              <div class="score-bar" style="--bar-w:{bar_pct}%;--bar-color:{color};width:0%;height:100%;background:{color};border-radius:99px;box-shadow:0 0 8px {color}55"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:5px">
              <span style="font-family:IBM Plex Mono,monospace;font-size:8px;color:rgba(255,255,255,0.15)">0.0</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:8px;color:rgba(255,255,255,0.15)">1.0</span>
            </div>
          </div>
          <style>
            .score-bar {{ animation: fillBar 1.2s cubic-bezier(0.4,0,0.2,1) forwards; animation-delay: 0.2s; }}
            @keyframes fillBar {{ from {{ width: 0% }} to {{ width: var(--bar-w) }} }}
          </style>
        </div>""", unsafe_allow_html=True)

    with mcol:
        st.markdown(f"""<div class="panel" style="height:100%;animation:fadeUp 0.6s ease both;animation-delay:0.4s">
          <div class="metric-grid" style="margin-bottom:14px">
            <div class="metric-cell">
              <div class="metric-key">Composite Score</div>
              <div class="metric-num" style="color:#60a5fa">{score:.4f}</div>
            </div>
            <div class="metric-cell">
              <div class="metric-key">P(Stable)</div>
              <div class="metric-num" style="color:#34d399">{stable:.4f}</div>
            </div>
            <div class="metric-cell">
              <div class="metric-key">Risk Flags</div>
              <div style="margin-top:8px">{flags_html}</div>
            </div>
          </div>
          <div style="background:#0a0b14;border:1px solid rgba(255,255,255,0.06);border-radius:3px;padding:14px 16px">
            <div style="font-family:IBM Plex Mono,monospace;font-size:9px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:rgba(0,122,255,0.55);margin-bottom:10px">Formulation Used</div>
            {pills_html}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="card-row">
      <div class="info-card">
        <div class="info-card-label">Grade Scale</div>
        <div class="info-card-body">A ≥0.80 Excellent · B ≥0.70 Good · C ≥0.60 Marginal · D ≥0.55 Poor · F &lt;0.55 Fail</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Composite Score</div>
        <div class="info-card-body">Weighted combination of aggregation, oxidation, and colloidal stability indicators — normalised 0–1.</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Disclaimer</div>
        <div class="info-card-body">Synthetic training data. Wet-lab validation required before making formulation decisions.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # ANALYTICS — SHAP · Feature Importance · pH Sweep
    # ════════════════════════════════════════════════════
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect x="10" y="2" width="4" height="10" rx="1" stroke="#007AFF" stroke-width="1.5"/><path d="M8 10L5 19H19L16 10" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><line x1="8" y1="6" x2="16" y2="6" stroke="#34d399" stroke-width="1" opacity="0.6"/><circle cx="12" cy="15" r="1.5" fill="#34d399" opacity="0.8"/></svg></div>
        <div>
          <div class="panel-title">Explainability & Sensitivity Analysis</div>
          <div class="panel-step">SHAP · Feature Importance · pH Sweep</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def clean_label(name):
        prefixes = ["buffer_","sugar_","surfactant_","amino_acid_","salt_","query_label_"]
        for p in prefixes:
            if name.startswith(p):
                return name.replace(p, "").replace("_", " ").title()
        return name.replace("_", " ").title()

    pid   = r["protein_id"]
    _match = _pf[_pf["protein_id"] == pid]
    prow  = _match.iloc[0] if len(_match) > 0 else _pf.iloc[0]
    vec   = _build_vector(prow, r["formulation"]).reshape(1, -1)

    tab1, tab2, tab3 = st.tabs(["SHAP Explanation", "Feature Importance", "pH Sensitivity"])

    # ── Tab 1: SHAP ──────────────────────────────────────
    with tab1:
        st.caption("Which features pushed the composite score up or down from the baseline?")
        with st.spinner("Computing SHAP values..."):
            explainer   = shap.TreeExplainer(_reg_model)
            shap_values = explainer.shap_values(vec)
            sv          = shap_values[0]
            base_val    = float(explainer.expected_value)

        indices  = np.argsort(np.abs(sv))[::-1][:12]
        top_feat = [_feature_cols[i] for i in indices]
        top_shap = [sv[i] for i in indices]
        labels   = [clean_label(f) for f in top_feat]
        colors   = ["#34d399" if v > 0 else "#f87171" for v in top_shap]

        fig_shap = go.Figure(go.Bar(
            x=top_shap, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in top_shap],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#e2e4ed"),
        ))
        fig_shap.add_vline(x=0, line_color="#8b949e", line_width=1)
        fig_shap.update_layout(
            paper_bgcolor="#0d0e18", plot_bgcolor="#0d0e18",
            font=dict(family="IBM Plex Sans", color="#e2e4ed"),
            height=380,
            xaxis=dict(title="SHAP value (impact on composite score)", showgrid=True, gridcolor="#1e2230", zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(t=10, b=40, l=10, r=80),
            annotations=[dict(
                x=0.01, y=1.04, xref="paper", yref="paper",
                text=f"Baseline: {base_val:.4f}  →  Prediction: {r['pred_composite_score']:.4f}",
                showarrow=False,
                font=dict(size=11, color="rgba(255,255,255,0.35)", family="IBM Plex Mono"),
            )]
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("""
        <div class="notice">
          <b>Green bars</b> = features that increase stability score.
          <b>Red bars</b> = features that reduce it. Length = magnitude of impact.
        </div>""", unsafe_allow_html=True)

    # ── Tab 2: Feature Importance ────────────────────────
    with tab2:
        st.caption("Overall importance of each feature across all predictions (XGBoost classifier)")
        importances = _cls_model.feature_importances_
        imp_df = pd.DataFrame({
            "feature":    [clean_label(f) for f in _feature_cols],
            "importance": importances,
        }).sort_values("importance", ascending=False).head(15)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["feature"], orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale=[[0,"#0a1628"],[0.5,"#1a5fb4"],[1,"#007AFF"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in imp_df["importance"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
        ))
        fig_imp.update_layout(
            paper_bgcolor="#0d0e18", plot_bgcolor="#0d0e18",
            font=dict(family="IBM Plex Sans", color="#e2e4ed"),
            height=400,
            xaxis=dict(title="Feature Importance (gain)", showgrid=True, gridcolor="#1e2230"),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(t=10, b=40, l=10, r=80),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Tab 3: pH Sensitivity ────────────────────────────
    with tab3:
        st.caption("How composite score changes across the full pH range (all other inputs held constant)")
        ph_values = ALLOWED["ph"]
        scores_ph, probas_ph = [], []
        for ph_val in ph_values:
            tf = r["formulation"].copy()
            tf["ph"] = ph_val
            tv = _build_vector(prow, tf).reshape(1, -1)
            scores_ph.append(float(_reg_model.predict(tv)[0]))
            probas_ph.append(float(_cls_model.predict_proba(tv)[0, 1]))

        current_ph = float(r["formulation"]["ph"])
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=ph_values, y=scores_ph, name="Composite Score",
            line=dict(color="#007AFF", width=2.5), mode="lines+markers",
            marker=dict(size=7, color="#007AFF"),
            hovertemplate="pH %{x}<br>Score: %{y:.4f}<extra></extra>",
        ))
        fig_ph.add_trace(go.Scatter(
            x=ph_values, y=probas_ph, name="P(Stable)",
            line=dict(color="#34d399", width=2, dash="dot"), mode="lines+markers",
            marker=dict(size=6, color="#34d399"),
            hovertemplate="pH %{x}<br>P(Stable): %{y:.4f}<extra></extra>",
        ))
        fig_ph.add_vline(
            x=current_ph, line_color="#fbbf24", line_dash="dash", line_width=1.5,
            annotation_text=f"Current pH {current_ph}",
            annotation_font=dict(color="#fbbf24", size=11, family="IBM Plex Mono"),
            annotation_position="top right",
        )
        for y_val, lbl, col in [
            (0.80,"A","rgba(52,211,153,0.08)"),
            (0.70,"B","rgba(163,230,53,0.06)"),
            (0.60,"C","rgba(251,191,36,0.06)"),
            (0.55,"D","rgba(249,115,22,0.06)"),
        ]:
            fig_ph.add_hrect(
                y0=y_val, y1=min(y_val+0.10, 1.05),
                fillcolor=col, line_width=0,
                annotation_text=f"Grade {lbl}",
                annotation_position="right",
                annotation_font=dict(size=10, color="rgba(255,255,255,0.25)", family="IBM Plex Mono"),
            )
        fig_ph.update_layout(
            paper_bgcolor="#0d0e18", plot_bgcolor="#0d0e18",
            font=dict(family="IBM Plex Sans", color="#e2e4ed"),
            height=360,
            xaxis=dict(title="Formulation pH", tickvals=ph_values, showgrid=True, gridcolor="#1e2230"),
            yaxis=dict(title="Score / Probability", range=[0,1.05], showgrid=True, gridcolor="#1e2230"),
            legend=dict(bgcolor="#0d0e18", bordercolor="#30363d", borderwidth=1),
            margin=dict(t=10, b=40, l=50, r=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ph, use_container_width=True)

        best_idx = int(np.argmax(scores_ph))
        best_ph  = ph_values[best_idx]
        best_sc  = scores_ph[best_idx]
        st.markdown(f"""
        <div class="notice">
          <b>Optimal pH for this formulation:</b> {best_ph}
          &nbsp;·&nbsp; Composite score: <b>{best_sc:.4f}</b>
          {"&nbsp;·&nbsp; ✓ Same as current" if best_ph == current_ph else f"&nbsp;·&nbsp; ⚠ Consider switching from pH {current_ph}"}
        </div>
        </div>""", unsafe_allow_html=True)


    # ── Scenario Simulation ──────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-badge"><svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2v-4M9 21H5a2 2 0 01-2-2v-4m0 0h18" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg></div>
        <div>
          <div class="panel-title">Scenario Simulation</div>
          <div class="panel-step">Swap one variable — see instant score impact</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sim_var = st.selectbox(
        "SELECT VARIABLE TO SWAP",
        ["buffer", "sugar", "surfactant", "amino_acid", "salt", "ph", "temperature_c", "protein_conc_mgmL"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    current_val = r["formulation"].get(sim_var)
    options = ALLOWED[sim_var]
    other_options = [o for o in options if str(o) != str(current_val)]

    if other_options:
        sim_results = []
        for opt in options:
            sim_form = r["formulation"].copy()
            sim_form[sim_var] = opt
            sim_pred = predict_single(r["protein_id"], sim_form)
            delta = sim_pred["pred_composite_score"] - r["pred_composite_score"]
            sim_results.append({
                "value": opt,
                "score": sim_pred["pred_composite_score"],
                "stable": sim_pred["pred_stable_proba"],
                "delta": delta,
                "is_current": str(opt) == str(current_val),
                "grade": sim_pred["stability_grade"],
            })

        sim_results.sort(key=lambda x: x["score"], reverse=True)

        grade_colors = {"A":"#34d399","B":"#a3e635","C":"#fbbf24","D":"#f97316","F":"#f87171"}

        rows = ""
        for sr in sim_results:
            gc = grade_colors.get(sr["grade"], "#e2e4ed")
            delta_color = "#34d399" if sr["delta"] > 0 else "#f87171" if sr["delta"] < 0 else "#e2e4ed"
            delta_str = f"+{sr['delta']:.4f}" if sr["delta"] >= 0 else f"{sr['delta']:.4f}"
            current_marker = " ← current" if sr["is_current"] else ""
            row_bg = "background:rgba(0,122,255,0.06)" if sr["is_current"] else ""
            rb = "border-bottom:1px solid rgba(255,255,255,0.05)"
            td = f"padding:12px 20px;font-family:IBM Plex Mono,monospace;font-size:12px;{rb}"
            rows += (
                f"<tr style='{row_bg}'>"
                f"<td style='{td};color:rgba(226,228,237,0.9)'>{sr['value']}"
                f"<span style='font-size:10px;color:rgba(0,122,255,0.6);margin-left:6px'>{current_marker}</span></td>"
                f"<td style='{td};text-align:right;color:#e2e4ed'>{sr['score']:.4f}</td>"
                f"<td style='{td};text-align:right;color:{delta_color};font-weight:600'>{delta_str}</td>"
                f"<td style='{td};text-align:right;color:#34d399'>{sr['stable']:.4f}</td>"
                f"<td style='{td};text-align:center'>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:600;"
                f"color:{gc};border:1px solid {gc};border-radius:3px;padding:1px 7px'>{sr['grade']}</span>"
                f"</td>"
                "</tr>"
            )

        th = "padding:10px 20px;font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(255,255,255,0.25);font-weight:500;border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap"
        header = (
            "<tr>"
            f"<th style='{th};text-align:left'>{sim_var.replace('_',' ').title()}</th>"
            f"<th style='{th};text-align:right'>Score</th>"
            f"<th style='{th};text-align:right'>Delta</th>"
            f"<th style='{th};text-align:right'>P(Stable)</th>"
            f"<th style='{th};text-align:center'>Grade</th>"
            "</tr>"
        )
        table_html = (
            "<div style='margin-top:12px;overflow-x:auto;-webkit-overflow-scrolling:touch;"
            "scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.12) transparent'>"
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead>" + header + "</thead>"
            "<tbody>" + rows + "</tbody>"
            "</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # Best alternative callout
        best = [s for s in sim_results if not s["is_current"]]
        if best:
            best = best[0]
            if best["delta"] > 0:
                st.markdown(f"""
                <div class="notice" style="margin-top:12px;border-left-color:rgba(52,211,153,0.5);background:rgba(52,211,153,0.05)">
                  <b>Best alternative:</b> Switching <b>{sim_var.replace('_',' ')}</b> from
                  <b>{current_val}</b> to <b>{best['value']}</b> improves composite score by
                  <b style="color:#34d399">{best['delta']:+.4f}</b> → Grade <b>{best['grade']}</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="notice" style="margin-top:12px">
                  <b>Current {sim_var.replace('_',' ')} is already optimal</b> for this protein.
                  No alternative improves the composite score.
                </div>""", unsafe_allow_html=True)

    # ── Bottom buttons ───────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Edit Conditions", use_container_width=True):
            goto(3); st.rerun()
    with c2:
        if st.button("+ New Assessment", type="primary", use_container_width=True):
            goto(1); st.rerun()
    with c3:
        pdf_bytes = generate_pdf_report(r)
        st.download_button(
            "⬇ Export PDF",
            data=pdf_bytes,
            file_name=f"{r['protein_id']}_stability_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.checkbox("Show top 10 recommendations for this protein"):
        with st.spinner("Evaluating 4,320 formulation combinations..."):
            from src.formulation_optimizer import optimize_formulation
            top, _ = optimize_formulation(
                r["protein_id"],
                target_temp=r["formulation"]["temperature_c"],
                target_conc=r["formulation"]["protein_conc_mgmL"],
                top_n=10, verbose=False
            )
        cols = ["rank","buffer","sugar","surfactant","amino_acid","salt","ph","pred_composite_score","pred_stable_proba"]
        top_show = top[cols].copy()

        def grade_color(score):
            if score >= 0.80: return "#34d399"
            elif score >= 0.70: return "#a3e635"
            elif score >= 0.60: return "#fbbf24"
            elif score >= 0.55: return "#f97316"
            else: return "#f87171"

        def grade_label(score):
            if score >= 0.80: return "A"
            elif score >= 0.70: return "B"
            elif score >= 0.60: return "C"
            elif score >= 0.55: return "D"
            else: return "F"

        rows_html = ""
        for _, row in top_show.iterrows():
            gc = grade_color(row["pred_composite_score"])
            gl = grade_label(row["pred_composite_score"])
            is_top = int(row['rank']) == 1
            row_bg = "background:rgba(0,122,255,0.04)" if is_top else ""
            rb = "border-bottom:1px solid rgba(255,255,255,0.06)"
            td = f"padding:13px 20px;white-space:nowrap;font-family:IBM Plex Mono,monospace;font-size:12px;color:rgba(226,228,237,0.85);{rb}"
            td_dim = f"padding:13px 20px;white-space:nowrap;font-family:IBM Plex Mono,monospace;font-size:11px;color:rgba(255,255,255,0.30);{rb}"
            rows_html += (
                f"<tr style='{row_bg}'>"
                f"<td style='{td_dim}'>{int(row['rank']):02d}</td>"
                f"<td style='{td}'>{row['buffer']}</td>"
                f"<td style='{td}'>{row['sugar']}</td>"
                f"<td style='{td}'>{row['surfactant']}</td>"
                f"<td style='{td}'>{row['amino_acid']}</td>"
                f"<td style='{td}'>{row['salt']}</td>"
                f"<td style='{td};text-align:right;color:rgba(255,255,255,0.50)'>{row['ph']}</td>"
                f"<td style='{td};text-align:right;color:#34d399'>{row['pred_stable_proba']:.3f}</td>"
                f"<td style='padding:13px 20px;text-align:center;{rb}'>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:600;"
                f"color:{gc};border:1px solid {gc};border-radius:3px;padding:1px 7px;opacity:0.85'>{gl}</span>"
                f"</td>"
                "</tr>"
            )

        th = "padding:10px 20px;font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(255,255,255,0.25);font-weight:500;white-space:nowrap;border-bottom:1px solid rgba(255,255,255,0.08)"
        header = (
            "<tr>"
            f"<th style='{th};text-align:left'>No.</th>"
            f"<th style='{th};text-align:left'>Buffer</th>"
            f"<th style='{th};text-align:left'>Sugar</th>"
            f"<th style='{th};text-align:left'>Surfactant</th>"
            f"<th style='{th};text-align:left'>Amino Acid</th>"
            f"<th style='{th};text-align:left'>Salt</th>"
            f"<th style='{th};text-align:right'>pH</th>"
            f"<th style='{th};text-align:right'>P(Stable)</th>"
            f"<th style='{th};text-align:center'>Grade</th>"
            "</tr>"
        )
        table_html = (
            "<div style='margin-top:12px;overflow-x:auto;-webkit-overflow-scrolling:touch;"
            "scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.12) transparent'>"
            "<table style='width:max-content;min-width:100%;border-collapse:collapse'>"
            "<thead>" + header + "</thead>"
            "<tbody>" + rows_html + "</tbody>"
            "</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button("⬇ Download CSV", top.to_csv(index=False),
            file_name=f"{r['protein_id']}_top10.csv", mime="text/csv")
