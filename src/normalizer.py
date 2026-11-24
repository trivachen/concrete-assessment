from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Normalizer:
    u: np.ndarray

    def normalize(self) -> np.ndarray:
        return (self.u - self.u.mean()) / self.u.std()

    def denormalize(self, v: np.ndarray) -> np.ndarray:
        return v * self.u.std() + self.u.mean()
