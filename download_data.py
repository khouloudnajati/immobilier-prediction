from sklearn.datasets import fetch_california_housing
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

housing = fetch_california_housing(as_frame=True)

df = housing.frame

df.to_csv(DATA_DIR / "california_housing.csv", index=False)

print("Dataset téléchargé !")
print("Dimensions :", df.shape)
print(df.head())