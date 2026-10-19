from pathlib import Path
from sklearn.datasets import fetch_openml

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

data = fetch_openml(data_id=42165, as_frame=True)
df = data.frame

df.to_csv(RAW / "train.csv", index=False)

print("Shape:", df.shape)
print("Saved to:", RAW / "train.csv")