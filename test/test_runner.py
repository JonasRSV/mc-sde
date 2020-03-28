import unittest
from sdepy.master import run
from sdepy.sde.ito import Ito1D
from sdepy.sde.exp import Exp1D
from sdepy.pdf.histogram import Simple1DHistogram
from sdepy.core import Job
from sdepy.video import make_time_1dx_distplot_video
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def drift(t, x):
    return 0


def diffusion(t, x):
    return 1


class TestRunner(unittest.TestCase):
    def test_raw_runner_ito_1d(self):
        sde = Ito1D(particles=10000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    sigma=1,
                    t0=0.0,
                    dt=1e-2,
                    T=1.0)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.RAW,
            settings={}
        )

        pdf = run(job, 2).distributions[-1]

        x = np.linspace(pdf.lower_bound, pdf.upper_bound, 1000)
        y = pdf(x)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="Approximation")
        sb.distplot(np.random.normal(0, 1, size=10000), label="True")
        plt.show()

    def test_raw_runner_exp_1d(self):
        sde = Exp1D(particles=10000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    lam=1,
                    t0=0.0,
                    dt=1e-2,
                    T=1.0)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.RAW,
            settings={}
        )

        pdf = run(job, 2).distributions[-1]

        x = np.linspace(pdf.lower_bound, pdf.upper_bound, 1000)
        y = pdf(x)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="Approximation")
        sb.distplot(np.random.normal(0, 1, size=10000), label="True")
        plt.show()

    def test_video_runner_ito1d(self):
        sde = Ito1D(particles=10000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    sigma=1,
                    t0=0.0,
                    dt=1e-2,
                    T=1.5)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.Video,
            settings={"steps_per_frame": 5}
        )

        result = run(job, 2).distributions

        make_time_1dx_distplot_video(result, fps=10, dt=1e-2, steps_per_frame=5)

    def test_video_runner_exp1d(self):
        sde = Exp1D(particles=10000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    lam=1,
                    t0=0.0,
                    dt=1e-2,
                    T=2.0)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.Video,
            settings={"steps_per_frame": 5}
        )

        result = run(job, 2).distributions

        make_time_1dx_distplot_video(result, fps=10, dt=1e-2, steps_per_frame=5)


if __name__ == '__main__':
    unittest.main()
