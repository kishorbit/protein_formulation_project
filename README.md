# PredStabio™
### AI-Powered Protein Formulation Stability Prediction Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-brightgreen)](https://xgboost.ai)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)

---

## Overview
PredStabio predicts the stability grade of biologic drug formulations using ML. Given a protein and excipient composition, it outputs a composite stability score, grade (A-F), SHAP explanation, and top-10 optimized formulation recommendations.

## Features
- Stability Prediction - XGBoost multi-output model scoring 4,320 formulation combinations
- Explainability - SHAP waterfall, Feature Importance, pH Sensitivity sweep
- Scenario Simulation - swap any excipient and see instant score delta
- Formulation Optimizer - top 10 ranked recommendations with grade badges
- PDF Export - one-click assessment report download

## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Custom CSS |
| ML Models | XGBoost, scikit-learn |
| Explainability | SHAP |
| PDF Generation | fpdf2 |

## Quick Start
```bash
git clone https://github.com/yourusername/predstabio.git
cd predstabio
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Docker
```bash
docker build -t predstabio .
docker run -p 8501:8501 predstabio
```

## Project Structure
predstabio/
├── streamlit_app.py
├── src/
│   ├── predict_api.py
│   └── formulation_optimizer.py
├── models/
├── data/
├── requirements.txt
└── Dockerfile
## Disclaimer
Synthetic training data. Wet-lab validation required before making formulation decisions.
