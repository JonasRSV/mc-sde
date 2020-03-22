import unittest
from sdepy.pdf import histogram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


class TestHistogram(unittest.TestCase):
    def test_fit_particles(self):
        particles = np.random.normal(0, 2, 100)
        # particles = np.random.exponential(scale=1, size=100)
        hist = histogram.Simple1DHistogram(particles=particles, bins=5)

        print(hist.mass)
        x = np.linspace(-8, 8, 100)
        y = hist(x)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="my histogram")
        sb.distplot(particles, label="seaborn histogram")
        plt.legend()
        plt.show()

    def test_merge_histograms(self):
        particles = np.random.normal(2, 1, 100000)
        h1 = histogram.Simple1DHistogram(particles=particles, bins=100)
        particles = np.random.normal(-2, 1, 100000)
        h2 = histogram.Simple1DHistogram(particles=particles, bins=200)

        h3 = h1.merge(h2)

        x = np.linspace(-8, 8, 100)
        y1 = h1(x)
        y2 = h2(x)
        y3 = h3(x)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, label="h1")
        plt.plot(x, y2, label="h2")
        plt.plot(x, y3, label="merged")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
