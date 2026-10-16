from pathlib import Path
import pandas as pd

RAW = Path("data/raw/train.csv")
PROCESSED = Path("data/processed/train_clean.csv")

FEATURES = ["Neighborhood", "GrLivArea", "BedroomAbvGr", "Bathrooms", "GarageCars"]
TARGET = "SalePrice"
OUTLIER_IDS = [524, 1299]

def load_and_clean(path=RAW) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

    df = df[~df["Id"].isin(OUTLIER_IDS)]

    return df[FEATURES + [TARGET]].reset_index(drop=True)


if __name__ == "__main__":
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    clean = load_and_clean()
    clean.to_csv(PROCESSED, index=False)
    print("Shape:", clean.shape)
    print("Saved to:", PROCESSED)
    print(clean.head())