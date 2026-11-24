from dataclasses import dataclass
from src.loss import LossCalculator
import numpy as np


@dataclass(frozen=True)
class Trainer:
    y: np.ndarray
    X: np.ndarray

    def initial_weights(self) -> np.ndarray:
        return np.random.rand(self.X.shape[1], 1)

    def epoch(
        self, w: np.ndarray, learning_rate: np.float32 = 0.0001
    ) -> tuple[np.float32, np.ndarray]:
        lc = LossCalculator(self.y, self.X)
        losses = lc.loss(w)
        delta = lc.loss_gradient(w)
        return losses, w - learning_rate * delta
