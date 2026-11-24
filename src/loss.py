from dataclasses import dataclass
from src.utils import compose
import numpy as np


@dataclass(frozen=True)
class LossCalculator:
    y: np.array
    X: np.array

    def loss(self, w: np.array) -> np.float32:
        y_tilde = self.X @ w
        f = compose(
            sum,
            lambda x: x**2,
            lambda y_tilde: self.y - y_tilde,
            lambda y_tilde: (
                y_tilde
                if y_tilde.shape == self.y.shape
                else y_tilde.reshape(self.y.shape)
            ),
        )
        return f(y_tilde)

    def loss_gradient(self, w: np.array) -> np.float32:
        m = self.X.shape[0]
        f = compose(
            lambda x: x * 2 / m,
            lambda x: self.X.T @ x,
            lambda x: x**2,
            lambda w: self.X @ w - self.y,
            lambda w: (
                w if w.shape == self.y.shape else w.reshape(self.y.shape)
            ),
        )
        return f(w)
