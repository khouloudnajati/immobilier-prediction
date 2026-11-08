"""Entraînement : exploration, split 70/15/15, comparaison de modèles, sauvegarde du pipeline."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42

# 1. Chargement et exploration
data = fetch_california_housing(as_frame=True)
df = data.frame
print("Dimensions :", df.shape)
print(df.isna().sum())
print(df.describe().T)

# Nettoyage : MedHouseVal est plafonné à 5.0 (valeurs censurées) -> on les retire
n_before = len(df)
df = df[df["MedHouseVal"] < 5.0].copy()
print(f"Lignes supprimées (cible plafonnée) : {n_before - len(df)}")

X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# 2. Split 70 / 15 / 15 (avant tout prétraitement => pas de fuite)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# 3. Pipelines : le prétraitement est ajusté UNIQUEMENT sur le train via fit()
def make_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

candidates = {
    "LinearRegression": make_pipe(LinearRegression()),
    "Ridge": make_pipe(Ridge(alpha=1.0)),
    "RandomForest": make_pipe(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": make_pipe(HistGradientBoostingRegressor(random_state=SEED)),
}

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

# 4. Comparaison sur la validation
val_results = {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    val_results[name] = metrics(y_val, pipe.predict(X_val))
    print(name, val_results[name])

best_name = min(val_results, key=lambda n: val_results[n]["RMSE"])
best = candidates[best_name]
print("Meilleur modèle (validation) :", best_name)

# 5. Évaluation finale UNE SEULE FOIS sur le test
test_metrics = metrics(y_test, best.predict(X_test))
print("Test :", test_metrics)

# 6. Sauvegarde
joblib.dump(best, "model/pipeline.joblib")
with open("model/metadata.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "features": list(X.columns),
        "validation": val_results,
        "test": test_metrics,
    }, f, indent=2)
print("Pipeline sauvegardé dans model/pipeline.joblib")
