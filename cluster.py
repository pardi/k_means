import numpy as np
from typing import Union


class Cluster:
    def __init__(self, dim: int = 2, name: str = "C0", col: str = "b"):
        self.name = name
        self.samples = []
        self.dim = dim
        self.mean = None
        self.col = col
        self.reset()

    def empty(self) -> None:
        self.samples = []

    def reset(self, init_sample: np.array = None) -> None:
        if init_sample is None:
            init_sample = np.zeros((self.dim, 1))

        self.mean = 0.5 * np.random.rand(self.dim, 1) + init_sample
        self.empty()

    def update_mean(self):
        if self.samples:
            self.mean = np.mean(self.samples, axis=0).reshape((self.dim, 1))
