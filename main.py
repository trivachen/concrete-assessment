from src.normalizer import Normalizer
from src.trainer import Trainer
from src.utils import compose
from pathlib import Path
import pandas as pd
import numpy as np


def process_xls(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(path)
    df = df[df[df.columns[-1]] < df[df.columns[-1]].quantile(0.95)]

    f = compose(
        lambda x: x.reshape(len(df), 1),
        lambda x: x.to_numpy(),
        lambda x: x[x.columns[-1]],
    )

    X = df[df.columns[:-1]].to_numpy()
    y = f(df)
    return y, X


def main() -> None:
    y, X = process_xls("data/Concrete_Data.xls")
    X_scaled = Normalizer(X).normalize()
    trainer = Trainer(y, X_scaled)
    w = trainer.initial_weights()
    losses, w = trainer.train(w)


if __name__ == "__main__":
    main()
