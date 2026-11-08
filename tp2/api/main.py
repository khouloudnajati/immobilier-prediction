"""API REST : uvicorn api.main:app --reload"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
pipeline = joblib.load(BASE / "model" / "pipeline.joblib")
metadata = json.loads((BASE / "model" / "metadata.json").read_text())
DB = BASE / "history.db"

with closing(sqlite3.connect(DB)) as c:
    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        inputs TEXT, price REAL)""")
    c.commit()

app = FastAPI(title="Estimation immobilière", version="1.0")


class House(BaseModel):
    MedInc: float = Field(..., gt=0, le=20, description="Revenu médian du quartier (x10k$)")
    HouseAge: float = Field(..., ge=0, le=100, description="Âge médian des logements")
    AveRooms: float = Field(..., gt=0, le=50, description="Nombre moyen de pièces")
    AveBedrms: float = Field(..., gt=0, le=20, description="Nombre moyen de chambres")
    Population: float = Field(..., gt=0, le=50000)
    AveOccup: float = Field(..., gt=0, le=50, description="Occupation moyenne")
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["best_model"]}


@app.get("/metrics")
def get_metrics():
    return metadata


@app.post("/predict")
def predict(house: House):
    df = pd.DataFrame([house.model_dump()])[metadata["features"]]
    price = float(pipeline.predict(df)[0]) * 100_000  # cible en centaines de milliers de $
    with closing(sqlite3.connect(DB)) as c:
        c.execute("INSERT INTO history (inputs, price) VALUES (?, ?)",
                  (json.dumps(house.model_dump()), price))
        c.commit()
    return {"estimated_price_usd": round(price, 2)}


@app.get("/history")
def history(limit: int = 20):
    with closing(sqlite3.connect(DB)) as c:
        rows = c.execute(
            "SELECT id, created_at, inputs, price FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"id": r[0], "date": r[1], "inputs": json.loads(r[2]), "price": r[3]} for r in rows]
