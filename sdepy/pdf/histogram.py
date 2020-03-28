import numpy as np

from sdepy.core import PDF
from collections import defaultdict

epsilon = 1e-5


class Simple1DHistogram(PDF):

    @property
    def mean(self) -> float:
        return self._mean

    def __init__(self, bins: int, *args, **kwargs):
        self.bins = bins

    @staticmethod
    def of(bins: int, mass: np.ndarray, intervals: list, lower_bound: np.float, upper_bound: np.float, mean: np.float):
        histogram = Simple1DHistogram(bins)
        histogram.mass = mass
        histogram.intervals = intervals
        histogram.lower_bound = lower_bound
        histogram.upper_bound = upper_bound
        histogram._mean = mean
        return histogram

    def fit(self, particles: np.ndarray):
        self._fit_particles(particles, self.bins)

    def _fit_particles(self, particles: np.ndarray, bins: int):
        self.lower_bound, self.upper_bound = particles.min(), particles.max()
        interval_sz = (self.upper_bound - self.lower_bound) / bins

        self.intervals = [(self.lower_bound + interval_sz * b, self.lower_bound + interval_sz * (b + 1))
                          for b in range(bins)]

        self.mass = np.zeros(bins)

        for i, (lb, ub) in enumerate(self.intervals):
            mask = (lb <= particles) & (particles < ub)
            self.mass[i] = mask.sum()

        self.mass[bins - 1] += (particles == self.upper_bound).sum()

        # Normalize
        if self.upper_bound == self.lower_bound:
            self.mass = self.mass / (self.mass.sum() + epsilon)
        else:
            self.mass = self.mass / (self.mass.sum() * (self.upper_bound - self.lower_bound) / bins)

        self._mean = particles.mean()

    def _merge_intervals(self, pdf: "Simple1DHistogram"):
        lower_bound = np.min([pdf.lower_bound, self.lower_bound])
        upper_bound = np.max([pdf.upper_bound, self.upper_bound])
        bins = np.max([pdf.bins, self.bins])

        pdf_importance_weight = 1 - pdf.bins / (pdf.bins + self.bins)
        self_importance_weight = 1 - self.bins / (pdf.bins + self.bins)

        interval_sz = (upper_bound - lower_bound) / bins
        intervals = [(lower_bound + interval_sz * b, lower_bound + interval_sz * (b + 1))
                     for b in range(bins)]

        def mass_intersection(min_bound: int, max_bound: int,
                              lb: float, ub: float,
                              mass: np.ndarray, intervals: np.ndarray):
            local_mass = 0.0
            for j in range(min_bound, max_bound):
                s_lb, s_ub = intervals[j]

                # Interval intersection
                if lb >= s_ub or s_lb >= ub:
                    percentage_overlap = 0.0
                else:
                    i_lb, i_ub = max([lb, s_lb]), min([ub, s_ub])
                    percentage_overlap = (i_ub - i_lb) / (s_ub - s_lb)

                local_mass += percentage_overlap * mass[j]
            return local_mass

        mass = np.zeros(bins)
        i_self_lower, i_self_upper = 0, 0
        i_pdf_lower, i_pdf_upper = 0, 0
        for i, (lb, ub) in enumerate(intervals):

            while i_self_upper < self.mass.size and self.intervals[i_self_upper][1] <= ub:
                i_self_upper += 1

            while i_pdf_upper < pdf.mass.size and pdf.intervals[i_pdf_upper][1] <= ub:
                i_pdf_upper += 1

            local_mass = mass_intersection(i_self_lower, min([i_self_upper + 1, self.mass.size]),
                                           lb, ub,
                                           self.mass, self.intervals) * self_importance_weight

            local_mass += mass_intersection(i_pdf_lower, min([i_pdf_upper + 1, pdf.mass.size]),
                                            lb, ub,
                                            pdf.mass, pdf.intervals) * pdf_importance_weight

            mass[i] = local_mass

            i_self_lower = i_self_upper
            i_pdf_lower = i_pdf_upper

        if upper_bound == lower_bound:
            mass = mass / (mass.sum() + epsilon)
        else:
            mass = mass / (mass.sum() * (upper_bound - lower_bound) / bins)

        return Simple1DHistogram.of(bins=bins,
                                    mass=mass, intervals=intervals,
                                    lower_bound=lower_bound, upper_bound=upper_bound,
                                    mean=self.mean * self_importance_weight + pdf.mean * pdf_importance_weight)

    def merge(self, pdf: "PDF"):

        m = defaultdict(
            default_factory=lambda:
            NotImplementedError(f"Merging {pdf.__class__} with {self.__class__} is not implemented"))

        m[self.__class__] = self._merge_intervals

        return m[pdf.__class__](pdf)

    def _binary_search_interval(self, lower_bound: int, upper_bound: int, value: np.float) -> np.float:
        mean = (lower_bound + upper_bound) // 2
        lb, ub = self.intervals[mean]

        tol = ((self.upper_bound - self.lower_bound) / self.bins) / 100
        if lb - tol <= value <= ub + tol:
            return self.mass[mean]
        elif value < lb:
            return self._binary_search_interval(lower_bound, mean, value)
        else:
            return self._binary_search_interval(mean, upper_bound, value)

    def __call__(self, particles: np.ndarray):
        result = []
        for particle in particles:
            if self.lower_bound <= particle <= self.upper_bound:
                result.append(self._binary_search_interval(0, self.mass.size, particle))
            else:
                result.append(0)

        return np.array(result)

    def preprocess(self):
        pass
