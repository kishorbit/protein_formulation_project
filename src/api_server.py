"""
FormulAI — Prediction API Server
Run: uvicorn src.api_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Load once at startup ───────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.predict_api import predict_single, predict_batch, ALLOWED, _pf, _excipient_df

app = FastAPI(
    title="FormulAI",
    description="Protein Formulation Stability Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────
class FormulationInput(BaseModel):
    protein_id:         str
    buffer:             str
    sugar:              str
    surfactant:         str
    ph:                 float
    temperature_c:      float
    protein_conc_mgmL:  float
    amino_acid:         Optional[str] = "none"
    salt:               Optional[str] = "none"
    buf_conc_mM:        Optional[float] = None
    sug_conc_mM:        Optional[float] = None
    sur_conc_mM:        Optional[float] = None
    aa_conc_mM:         Optional[float] = None
    salt_conc_mM:       Optional[float] = None

class BatchInput(BaseModel):
    protein_id:    str
    formulations:  List[FormulationInput]

class RecommendInput(BaseModel):
    protein_id:        str
    temperature_c:     Optional[int] = 25
    protein_conc_mgmL: Optional[int] = 10
    top_n:             Optional[int] = 5

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "FormulAI", "version": "1.0.0", "status": "ok"}

@app.get("/proteins")
def list_proteins():
    """List all proteins available for prediction."""
    return {
        "proteins": _pf[["protein_id","query_label","isoelectric_point",
                          "instability_index"]].to_dict(orient="records")
    }

@app.get("/excipients")
def list_excipients():
    """List all allowed excipients by class."""
    return _excipient_df.groupby("class").apply(
        lambda g: g[["name","conc_min_mM","conc_max_mM"]].to_dict(orient="records")
    ).to_dict()

@app.get("/allowed")
def allowed_values():
    """Return all allowed values for formulation fields."""
    return ALLOWED

@app.post("/predict")
def predict(inp: FormulationInput):
    """Score a single formulation."""
    formulation = inp.model_dump()
    protein_id  = formulation.pop("protein_id")
    try:
        result = predict_single(protein_id, formulation)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/predict/batch")
def predict_batch_route(inp: BatchInput):
    """Score multiple formulations for one protein."""
    rows = [f.model_dump() for f in inp.formulations]
    for r in rows:
        r.pop("protein_id", None)
    df = pd.DataFrame(rows)
    try:
        result = predict_batch(inp.protein_id, df)
        return result.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/recommend")
def recommend_route(inp: RecommendInput):
    """Return top formulations for a protein."""
    from src.formulation_optimizer import optimize_formulation
    try:
        top, _ = optimize_formulation(
            inp.protein_id,
            target_temp=inp.temperature_c,
            target_conc=inp.protein_conc_mgmL,
            top_n=inp.top_n,
            verbose=False
        )
        return top.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "proteins_loaded": len(_pf)}
