# TP2 – Prédiction du prix d'un bien immobilier

## Installation
    python -m venv venv && source venv/bin/activate   # Windows : venv\Scripts\activate
    pip install -r requirements.txt

## Exécution (3 terminaux ou étapes)
1. Entraîner le modèle :   `python train.py`
2. Lancer l'API :          `uvicorn api.main:app --reload`   (docs : http://127.0.0.1:8000/docs)
3. Lancer l'interface :    `streamlit run frontend/app.py`

## Structure
    train.py            exploration, split 70/15/15, comparaison, sauvegarde
    model/              pipeline.joblib + metadata.json
    api/main.py         FastAPI + Pydantic + historique SQLite
    frontend/app.py     interface Streamlit
