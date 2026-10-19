from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from data_prep import load_and_clean, TARGET

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme()


def main():
    df = load_and_clean()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax[0])
    ax[0].set_title("SalePrice")
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=ax[1])
    ax[1].set_title("log1p(SalePrice)")
    fig.tight_layout()
    fig.savefig(FIG / "target_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="GrLivArea", y=TARGET, alpha=0.6, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "size_vs_price.png", dpi=120)
    plt.close(fig)

    imp = json.loads((ROOT / "results" / "metrics.json").read_text())
    s = pd.Series(imp["feature_importance_top10"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    s.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance (top 10)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=120)
    plt.close(fig)

    print("Figures saved to", FIG)


if __name__ == "__main__":
    main()
