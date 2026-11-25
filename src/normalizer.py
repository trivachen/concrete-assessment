from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Normalizer:
    x: np.ndarray

    def normalize(self) -> np.ndarray:
        return (self.x - self.x.mean()) / self.x.std()

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        return y * self.x.std() + self.x.mean()
