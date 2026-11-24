from dataclasses import dataclass
from src.loss import LossCalculator
import numpy as np


@dataclass(frozen=True)
class Trainer:
    y: np.array
    X: np.array

    def initial_weights(self) -> np.array:
        return np.random.rand(self.X.shape[0], 1)

    def y_tilde(self, w: np.array) -> np.array:
        return self.X @ w

    def epoch(self, w: np.array) -> tuple[np.float32 | np.array]:
        lc = LossCalculator(self.y, self.X)
        losses = lc.loss(w)
        delta = lc.loss_gradient(w)
        return losses, w - delta
