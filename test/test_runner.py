import unittest
from sdepy.master import run
from sdepy.sde.ito import Ito1D
from sdepy.sde.exp import Exp1D
from sdepy.pdf.histogram import Simple1DHistogram
from sdepy.core import Job
from sdepy.video import make_time_1dx_distplot_video, three_dee_plot
from sdepy.pdf.normal import Normal
from sdepy.metrics import expected_1d, variance_1d, relative_entropy_1d
from sdepy.video import make_time_1dx_distplot_video
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time

# population sz is size of entire population
# probability of transmission is p("healty meet sick" -> healthy gets sick")
# rate of interaction is the number of individuals each individual meets each day

population_sz = 1500000
probability_of_transmission = 0.1
rate_of_interaction = 3  # How many people each individual meets per day
rate_of_recovery = 14  # How many days to recover

has_been_sick = None


def covid_drift(t, x):
    global has_been_sick
    if has_been_sick is None:
        has_been_sick = np.zeros_like(x, dtype=np.float64)

    recovering = x * (1.0 / rate_of_recovery)
    has_been_sick = has_been_sick + recovering * 0.1

    return -recovering


def covid_diffusion(t, x):
    global has_been_sick
    if has_been_sick is None:
        has_been_sick = np.zeros_like(x)

    p_sick = x / population_sz
    p_healthy = 1 - p_sick
    p_person_meet_sick = 1 - np.float_power(p_healthy, rate_of_interaction)
    p_person_not_immune = 1 - (has_been_sick / population_sz)

    p_person_get_sick = p_person_meet_sick * probability_of_transmission * p_person_not_immune

    expected_infections_per_time_unit = p_person_get_sick * np.maximum(population_sz - x, 0)

    return expected_infections_per_time_unit


def _fitting_loop(particles, bins):
    lower_bound, upper_bound = particles.min(), particles.max()
    interval_sz = (upper_bound - lower_bound) / bins

    intervals = [(lower_bound + interval_sz * b, lower_bound + interval_sz * (b + 1))
                 for b in range(bins)]

    mass = np.zeros(bins)

    for i, (lb, ub) in enumerate(intervals):
        mask = (lb <= particles) & (particles < ub)
        mass[i] = mask.sum()

    mass[bins - 1] += (particles == upper_bound).sum()


def drift(t, x):
    return 0


def diffusion(t, x):
    return 1


class TestRunner(unittest.TestCase):
    def test_raw_runner_ito_1d(self):
        T = 0.5
        sde = Ito1D(particles=30000,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    sigma=1,
                    t0=0.0,
                    dt=1e-2,
                    T=T)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=100),
            mode=Job.RAW,
            settings={}
        )

        result, execution_time = run(job, processes=3)
        pdf = result.distributions[-1]

        print(f"Parallel Execution Time: {execution_time}")

        x = np.linspace(pdf.lower_bound, pdf.upper_bound, 1000)
        y = pdf(x)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="Approximation")
        sb.distplot(np.random.normal(0, np.sqrt(T), size=10000), label="True")
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

        result, execution_time = run(job, 2)
        pdf = result.distributions[-1]

        print(f"Parallel Execution Time: {execution_time}")

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

        result, _ = run(job, 2)
        result = result.distributions

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

        result, _ = run(job, 2)
        result = result.distributions

        make_time_1dx_distplot_video(result, fps=10, dt=1e-2, steps_per_frame=5)

    def test_performance_1(self):
        T = 1
        dt = 1e-1
        processes = 4
        N = int(1e6)  # int(1e8 / 6)
        bins = 50
        sde = Ito1D(particles=N,
                    x0=0.0,
                    drift=drift,
                    diffusion=diffusion,
                    sigma=1,
                    t0=0.0,
                    dt=dt,
                    T=T)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=bins),
            mode=Job.RAW,
            settings={}
        )

        result, execution_time = run(job, processes=processes)

        pdf = result.distributions[-1]

        print(f"Parallel Execution Time: {execution_time}")

        x = np.linspace(pdf.lower_bound, pdf.upper_bound, 10000)
        y = pdf(x)

        normal = Normal(mu=0, sigma=np.sqrt(T))

        print(
            f"Expected -- approximation: {expected_1d(domain=x, pdf=pdf)} | true: {expected_1d(domain=x, pdf=normal)}")
        print(
            f"Variance -- approximation: {variance_1d(domain=x, pdf=pdf)} | true: {variance_1d(domain=x, pdf=normal)}")
        print(f"Relative Entropy: {relative_entropy_1d(domain=x, p=normal, q=pdf)}")

        plt.figure(figsize=(10, 6))
        plt.title(f"T={T} dt={dt} N={N * processes} B={bins}", fontsize=20)
        plt.plot(x, y, label="Approximation")
        p = normal(x)
        sb.lineplot(x, p, label="True")
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig("../images/many-bins-many-particles.png")
        plt.show()

    def test_plot_results(self):
        high_n_high_t = [124, 56, 38.8, 32, 28.2, 24, 20.5]  # n = 2 * 1e5 # Steps = 1e4
        low_n_high_t = [29, 16.0, 13.3, 13.2, 13.3, 12.4, 11.1]  # n = 7 * 1e3 # steps = 1e5
        high_n_low_t = [60.3, 36.6, 29.4, 27.14, 22.4, 22.3, 21.4]  # n = 1e8 # steps = 10

        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.title("N=200 000 - Steps=10 000", fontsize=16)
        x = np.arange(1, 8)
        sb.lineplot(x, high_n_high_t, label="Measured")
        sb.lineplot(x, 124 / x, label="Optimal")
        plt.legend(fontsize=14)
        plt.xlabel("P", fontsize=14)
        plt.ylabel("Seconds", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplot(3, 1, 2)
        plt.title("N=7000 - Steps=100 000", fontsize=16)
        x = np.arange(1, 8)
        sb.lineplot(x, low_n_high_t, label="Measured")
        sb.lineplot(x, 29 / x, label="Optimal")
        plt.legend(fontsize=14)
        plt.xlabel("P", fontsize=14)
        plt.ylabel("Seconds", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplot(3, 1, 3)
        plt.title("N=100 000 000 - Steps=10", fontsize=16)
        x = np.arange(1, 8)
        sb.lineplot(x, high_n_low_t, label="Measured")
        sb.lineplot(x, 60.3 / x, label="Optimal")
        plt.legend(fontsize=14)
        plt.xlabel("P", fontsize=14)
        plt.ylabel("Seconds", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig("../images/performance.png", bbox_inches="tight")
        plt.show()

    def test_show_by_tests_are_off(self):
        addition_times = [1e-6, 1e-6, 1.4 * 1e-6, 5.6 * 1e-6, 1e-4, 1e-3, 1e-2, 1.5 * 1e-1, 1.7]
        leq_times_and_times = [2.7 * 1e-6, 2.8 * 1e-6, 4.3 * 1e-6, 8.9 * 1e-6, 1e-4, 1e-3, 1.7 * 1e-2, 1.7 * 1e-1]
        max_times = [3.6 * 1e-6, 3.7 * 1e-6, 3.8 * 1e-6, 1.9 * 1e-5, 4.2 * 1e-5, 4.4 * 1e-4, 6.2 * 1e-3, 6.2 * 1e-2,
                     6.4 * 1e-1]
        mean_times = [7.0 * 1e-6, 7.1 * 1e-6, 3.0 * 1e-5, 1.0 * 1e-5, 4.0 * 1e-5, 3.6 * 1e-4, 5.2 * 1e-3, 4.7 * 1e-2,
                      4.8 * 1e-1]
        mul_times = [1.1 * 1e-6, 1.3 * 1e-6, 1.5 * 1e-6, 5.6 * 1e-6, 7.1 * 1e-5, 2.0 * 1e-3, 1.6 * 1e-2, 1.5 * 1e-1]
        random_nrs = [6.9 * 1e-6, 1.0 * 1e-5, 4.3 * 1e-5, 3.8 * 1e-4, 3.5 * 1e-3, 2.8 * 1e-2, 3.0 * 1e-1]
        repeating = [5.3 * 1e-6, 5.8 * 1e-6, 8.6 * 1e-6, 3.5 * 1e-5, 4.0 * 1e-4, 3.0 * 1e-3, 3.0 * 1e-2, 3.2 * 1 - 1]
        broad_casting = [1.8 * 1e-6, 3.6 * 1e-6, 2.4 * 1e-6, 7.1 * 1e-6, 1.5 * 1e-4, 1.3 * 1e-3, 1.6 * 1e-2, 1.5 * 1e-1]
        accessing = [8.0 * 1e-7, 1.2 * 1e-6, 5.0 * 1e-6, 5.4 * 1e-5, 6.4 * 1e-4, 5.5 * 1e-3, 5.6 * 1e-2, 4.6 * 1e-1]

        x = np.logspace(1, 10, 10)

        ops = [
            (addition_times, "addition"),
            (leq_times_and_times, "cmp + logical"),
            (max_times, "max"),
            (mean_times, "mean"),
            (mul_times, "multiplication"),
            (random_nrs, "random numbers"),
            (repeating, "repeating"),
            (broad_casting, "broad casting"),
            (accessing, "access")
        ]

        plt.figure(figsize=(10, 6))
        plt.title("Numpy OP's performance", fontsize=16)
        for data, label in ops:
            _x = x[:len(data)]
            sb.lineplot(_x, data, label=label)

        plt.legend(fontsize=14)
        plt.xlabel("Data size", fontsize=15)
        plt.ylabel("Seconds", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.yscale("log")
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig("../images/numpy_ops.png", bbox_inches="tight")
        plt.show()

    def test_fitting_loop_complexity(self):
        outside_mpi = [(12.46, int(1e8 / 1)),
                       (6.39, int(1e8 / 2)),
                       (4.2, int(1e8 / 3)),
                       (3.0, int(1e8 / 4)),
                       (2.5, int(1e8 / 5)),
                       (2.4, int(1e8 / 6)),
                       (1.84, int(1e8 / 7))]

        inside_mpi = [(11.67, int(1e8 / 1)),
                      (7.8, int(1e8 / 2)),
                      (7.7, int(1e8 / 3)),
                      (6.9, int(1e8 / 4)),
                      (5.0, int(1e8 / 5)),
                      (6.7, int(1e8 / 6)),
                      (6.9, int(1e8 / 7))]

        """
        #for N in np.logspace(1, 8, 8):
            #particles = np.random.normal(0, 1, size=int(N))
        """

        bins = 50
        processes = 1
        N = int(1e8 / processes)
        particles = np.random.normal(0, 1, size=int(N))
        timestamp = time.time()
        _fitting_loop(particles, bins)
        print(f"Fitting {N} took", time.time() - timestamp)

        """

        outsidex, outsidey = zip(*outside_mpi)
        insidex, insidey = zip(*inside_mpi)

        plt.figure(figsize=(10, 6))
        plt.title("fit (inside / outside) mpi", fontsize=16)
        sb.lineplot(np.arange(1, 8), list(outsidex), label="outside - mpi")
        sb.lineplot(np.arange(1, 8), list(insidex), label="inside - mpi")
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Processes", fontsize=14)
        plt.ylabel("Seconds", fontsize=14)
        plt.tight_layout()
        plt.savefig("../images/fit-performance.png", bbox_inches="tight")
        plt.show()
        """

    def test_cool_accuracy_plot(self):
        T = 1
        dt = 1e-2
        processes = 2

        dim = (4, 4)
        _N = np.logspace(3, 6, 4)
        _bins = np.array([5, 50, 200, 500])  # , 200, 500])

        X, Y = np.meshgrid(_N, _bins)

        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        results = []
        for n, b in Z:
            N = int(n / processes)
            bins = int(b)

            sde = Ito1D(particles=N,
                        x0=0.0,
                        drift=drift,
                        diffusion=diffusion,
                        sigma=1,
                        t0=0.0,
                        dt=dt,
                        T=T)

            job = Job(
                sde=sde,
                pdf=Simple1DHistogram(bins=bins),
                mode=Job.RAW,
                settings={}
            )

            result, execution_time = run(job, processes=processes)

            pdf = result.distributions[-1]

            x = np.linspace(pdf.lower_bound, pdf.upper_bound, 10000)

            normal = Normal(mu=0, sigma=np.sqrt(T))
            kl_div = relative_entropy_1d(domain=x, p=normal, q=pdf)

            print(f"Parallel Execution Time: {execution_time} -- {n} -- {bins} --  {kl_div}")

            results.append(kl_div)

        results = np.array(results).reshape(*dim)

        plt.figure(figsize=(10, 6))
        plt.title(f"KL-divergence heatmap", fontsize=20)
        print("results", results)
        ax = sb.heatmap(results, annot=True, yticklabels=_bins, xticklabels=_N)
        plt.ylabel("Bins", fontsize=15)
        plt.xlabel("N", fontsize=15)
        plt.tight_layout()
        plt.savefig("../images/kl-heatmap.png", bbox_inches="tight")
        plt.show()

    def test_cool_sparseness_plot(self):
        dt = 1e-1
        processes = 2

        _N = 100000
        bins = 50

        _T = [1, 10, 100, 1000]  # 1000, 10000]

        results = []
        for T in _T:
            N = int(_N / processes)

            sde = Ito1D(particles=N,
                        x0=0.0,
                        drift=drift,
                        diffusion=diffusion,
                        sigma=1,
                        t0=0.0,
                        dt=dt,
                        T=T)

            job = Job(
                sde=sde,
                pdf=Simple1DHistogram(bins=bins),
                mode=Job.RAW,
                settings={}
            )

            result, execution_time = run(job, processes=processes)

            pdf = result.distributions[-1]

            x = np.linspace(-np.sqrt(T) * 3, np.sqrt(T) * 3, 100000)

            normal = Normal(mu=0, sigma=np.sqrt(T))
            kl_div = relative_entropy_1d(domain=x, p=normal, q=pdf)

            print(f"Parallel Execution Time: {execution_time} -- {_N} -- {bins} --  {kl_div}")

            results.append(kl_div)

        results = np.array(results)

        plt.figure(figsize=(10, 6))
        plt.title(f"KL-divergence sparseness effect", fontsize=20)
        print("results", results)
        ax = sb.lineplot(_T, results)
        plt.ylabel("KL-divergence", fontsize=15)
        plt.xlabel("T", fontsize=15)
        plt.tight_layout()
        # plt.savefig("../images/kl-heatmap.png", bbox_inches="tight")
        plt.show()

    def test_complexities(self):
        sampling_time_min_1e6 = [
            0.4,
            0.45,
            0.51,
            0.55,
            0.55,
            0.70,
        ]

        sampling_time_max_1e6 = [
            0.4,
            0.46,
            0.53,
            0.57,
            0.81,
            0.78,

        ]  # 1e7 samples

        fitting_time_min_1e6 = [
            0.102,
            0.141,
            0.20,
            0.14,
            0.19,
            0.33,
        ]  # 1e7

        fitting_time_max_1e6 = [
            0.102,
            0.144,
            0.23,
            0.25,
            0.21,
            0.40
        ]  # 1e7

        sampling_time_min_1e7 = [
            4.51,
            5.0,
            5.8,
            5.6,
            5.6,
            5.9,

        ]  # 1e7 samples

        sampling_time_max_1e7 = [
            4.51,
            5.0,
            5.8,
            5.6,
            7.71,
            8.5,
        ]  # 1e7 samples

        fitting_time_min_1e7 = [
            1.27,
            1.81,
            2.4,
            2.1,
            2.3,
            2.2,
        ]  # 1e7

        fitting_time_max_1e7 = [
            1.27,
            1.81,
            2.4,
            2.1,
            2.3,
            3.6,
        ]  # 1e7

        sampling_time_min_1e8 = [
            21.48,
            25.411,
            30.35,
            28.41,
            29.24,
            40.13,
        ]  # 1e8 samples

        sampling_time_max_1e8 = [
            21.48,
            25.418,
            30.40,
            38.31,
            42.5,
            46.9,
        ]  # 1e8 samples

        fitting_time_min_1e8 = [
            6.62,
            9.28,
            11.81,
            9.77,
            9.07,
            19.68,
        ]  # 1e8

        fitting_time_max_1e8 = [
            6.62,
            9.65,
            12.32,
            11.8,
            15.3,
            23.52,
        ]  # 1e8

        x = np.arange(1, 7)

        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.title("1e6 particles", fontsize=15)
        plt.plot(x, fitting_time_min_1e6, label="fitting time min")
        plt.plot(x, fitting_time_max_1e6, label="fitting time max")
        plt.plot(x, sampling_time_min_1e6, label="sampling time min")
        plt.plot(x, sampling_time_max_1e6, label="sampling time max")
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Processes", fontsize=12)
        plt.ylabel("Seconds", fontsize=12)

        plt.subplot(3, 1, 2)
        plt.title("1e7 particles", fontsize=15)
        plt.plot(x, fitting_time_min_1e7, label="fitting time min")
        plt.plot(x, fitting_time_max_1e7, label="fitting time max")
        plt.plot(x, sampling_time_min_1e7, label="sampling time min")
        plt.plot(x, sampling_time_max_1e7, label="sampling time max")
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Processes", fontsize=12)
        plt.ylabel("Seconds", fontsize=12)

        plt.subplot(3, 1, 3)
        plt.title("1e8 particles", fontsize=15)
        plt.plot(x, fitting_time_min_1e8, label="fitting time min")
        plt.plot(x, fitting_time_max_1e8, label="fitting time max")
        plt.plot(x, sampling_time_min_1e8, label="sampling time min")
        plt.plot(x, sampling_time_max_1e8, label="sampling time max")
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Processes", fontsize=12)
        plt.ylabel("Seconds", fontsize=12)

        plt.tight_layout()
        plt.savefig("../images/many-particles-odd-performance-drops.png", bbox_inches="tight")
        plt.show()

    def test_virus_model(self):
        global has_been_sick

        has_been_sick = None

        T = 200
        dt = 1e-1
        sde = Exp1D(particles=50000,
                    x0=1.0,
                    drift=covid_drift,
                    diffusion=covid_diffusion,
                    lam=1.0,
                    t0=0.0,
                    dt=dt,
                    T=T)

        job = Job(
            sde=sde,
            pdf=Simple1DHistogram(bins=50),
            mode=Job.Video,
            settings={"steps_per_frame": 5}
        )

        result, execution_time = run(job, processes=4)

        tdplot = make_time_1dx_distplot_video(result.distributions, fps=10, dt=dt, steps_per_frame=5, save=True)
        tdplot = three_dee_plot(result.distributions, dt=dt, steps_per_frame=5)
        plt.show()


if __name__ == '__main__':
    unittest.main()
