import numpy as np

from sdepy.core import PDF
from collections import defaultdict

epsilon = 1e-5


class Normal(PDF):

    @property
    def mean(self) -> float:
        return self._mean

    def __init__(self, mu: float, sigma: float, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma

    def fit(self, particles: np.ndarray):
        self.mu = particles.mean()
        self.sigma = np.sqrt(particles.var())

    def merge(self, pdf: "PDF"):
        m = defaultdict(
            default_factory=lambda:
            NotImplementedError(f"Merging {pdf.__class__} with {self.__class__} is not implemented"))

        return m[pdf.__class__](pdf)

    def __call__(self, particles: np.ndarray):
        sigma_sq = np.square(self.sigma)
        normalizing_constant = np.sqrt(2 * np.pi * sigma_sq)
        return np.exp((-1 / 2) * np.square(particles - self.mu) / sigma_sq) / normalizing_constant

    def preprocess(self):
        pass
