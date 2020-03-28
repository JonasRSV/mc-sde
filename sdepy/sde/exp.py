from sdepy.core import SDE
import numpy as np
from typing import Callable, Tuple


class Exp1D(SDE):

    @property
    def steps(self) -> int:
        return int((self.T - self.t) / self.dt)

    def __init__(self, particles: int,
                 x0: np.ndarray,
                 drift: Callable[[np.float64, np.float64], np.ndarray],
                 diffusion: Callable[[np.float64, np.float64], np.ndarray],
                 lam=None,
                 t0=0.0,
                 dt=1e-1,
                 T=1.0):
        if lam is None:
            raise Exception("Lam SDE missing argument lam")

        self.particles = particles
        self.x = x0
        self.drift = drift
        self.diffusion = diffusion
        self.lam = lam
        self.t = t0
        self.dt = dt
        self.T = T

    def preprocess(self):
        self.x = np.repeat(self.x, self.particles)

    def euler_maruyama(self) -> Tuple[np.float64, np.ndarray]:
        dL = np.random.exponential(1 / self.lam, size=self.particles) * self.dt

        self.x += self.drift(self.t, self.x) * self.dt + self.diffusion(self.t, self.x) * dL
        self.t += self.dt

        return self.t, self.x

    def step(self) -> Tuple[np.float64, np.ndarray]:
        return self.euler_maruyama()

    def stop(self) -> bool:
        return self.t >= self.T

