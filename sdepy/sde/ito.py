from sdepy.core import SDE
import numpy as np
from typing import Callable, Tuple


class Ito1D(SDE):

    def __init__(self, particles: int,
                 x0: np.ndarray,
                 drift: Callable[[np.float64, np.float64], np.ndarray],
                 diffusion: Callable[[np.float64, np.float64], np.ndarray],
                 sigma=None,
                 t0=0.0,
                 dt=1e-1,
                 T=1.0):
        if sigma is None:
            raise Exception("Ito1D SDE missing argument sigma")

        self.particles = particles
        self.x = x0
        self.drift = drift
        self.diffusion = diffusion
        self.sigma = sigma
        self.t = t0
        self.dt = dt
        self.T = T

    def preprocess(self):
        self.x = np.repeat(self.x, self.particles)

    def euler_maruyama(self) -> Tuple[np.float64, np.ndarray]:
        dW = np.random.normal(0, self.sigma, size=self.particles) * np.sqrt(self.dt)

        self.x += self.drift(self.t, self.x) * self.dt + self.diffusion(self.t, self.x) * dW
        self.t += self.dt

        return self.t, self.x

    def step(self) -> Tuple[np.float64, np.ndarray]:
        return self.euler_maruyama()

    def stop(self) -> bool:
        return self.t >= self.T

