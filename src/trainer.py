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
        self, w: np.ndarray, learning_rate: np.float64 = 0.0001
    ) -> tuple[np.float64, np.ndarray]:
        lc = LossCalculator(self.y, self.X)
        losses = lc.loss(w)
        delta = lc.loss_gradient(w)
        return losses, w - learning_rate * delta

    def train(
        self,
        w: np.ndarray,
        epochs: int = 10000,
        learning_rate: np.float64 = 0.0001,
    ) -> tuple[np.float64, np.ndarray]:
        for _ in range(epochs):
            losses, w = self.epoch(w, learning_rate)
        return losses, w
