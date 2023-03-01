import numpy as np


class Sampler:
    def __init__(self, num_distributions: int = 1, dim: int = 2):
        self.means = None
        self.stds = None
        self.num_distr = num_distributions
        self.dim = dim
        self._reset()

    def _reset(self):
        self.means = 1.0 * np.random.rand(self.num_distr, self.dim) - 0.5
        self.stds = .1 * np.random.rand(self.num_distr, 1)

    def __call__(self, *args, **kwargs) -> (np.array, int):

        while True:
            idx_distribution = np.random.choice(self.num_distr)

            sample = np.zeros((self.dim, 1))

            for idx_dim in range(self.dim):
                sample[idx_dim] = np.random.normal(self.means[idx_distribution][idx_dim], self.stds[idx_distribution])

            yield sample, idx_distribution
