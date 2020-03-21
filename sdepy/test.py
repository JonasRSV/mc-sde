import unittest
from sdepy import run
from ito import Ito1D


def drift(t, x):
    return 3


def diffusion(t, x):
    return 1


class TestRunner(unittest.TestCase):
    def test_runner(self):
        sde = Ito1D(particles=10,
                  x0=0.0,
                  drift=drift,
                  diffusion=diffusion,
                  sigma=1,
                  t0=0.0,
                  dt=1e-1,
                  T=1.0)

        run(sde, 2)


if __name__ == '__main__':
    unittest.main()
