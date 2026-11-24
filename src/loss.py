from dataclasses import dataclass
from src.utils import compose
import numpy as np


@dataclass(frozen=True)
class LossCalculator:
    y: np.ndarray
    X: np.ndarray

    def loss(self, w: np.ndarray) -> np.float64:
        f = compose(
            np.mean,
            lambda x: x**2,
            lambda y_tilde: self.y - y_tilde,
        )
        return f(self.X @ w)

    def loss_gradient(self, w: np.ndarray) -> np.float64:
        f = compose(
            lambda x: x * 2 / self.X.shape[0],
            lambda x: self.X.T @ x,
            lambda w: self.X @ w - self.y,
        )
        return f(w)
