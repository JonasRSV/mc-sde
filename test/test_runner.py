import unittest
from sdepy.master import run
from sdepy.sde.ito import Ito1D
from sdepy.pdf.histogram import Simple1DHistogram
from sdepy.core import Job
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def drift(t, x):
    return 1


def diffusion(t, x):
    return 1


class TestRunner(unittest.TestCase):
    def test_runner(self):
        sde = Ito1D(particles=10000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    sigma=1,
                    t0=0.0,
                    dt=1e-2,
                    T=2.0)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.RAW
        )

        pdf = run(job, 2)

        x = np.linspace(pdf.lower_bound, pdf.upper_bound, 1000)
        y = pdf(x)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="Approximation")
        sb.distplot(np.random.normal(0, 1, size=10000), label="True")
        plt.show()


if __name__ == '__main__':
    unittest.main()
