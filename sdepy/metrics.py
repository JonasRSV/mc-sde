from sdepy.core import PDF
import numpy as np

epsilon = 1e-15


def expected_1d(domain: np.ndarray, pdf: PDF):
    delta = domain[1] - domain[0]
    return (domain * pdf(domain) * delta).sum()


def variance_1d(domain: np.ndarray, pdf: PDF):
    delta = domain[1] - domain[0]

    mean = expected_1d(domain, pdf)
    central_domain = np.square(domain - mean)

    return (central_domain * pdf(domain) * delta).sum()


def relative_entropy_1d(domain: np.ndarray, p: PDF, q: PDF):
    delta = domain[1] - domain[0]
    return -(delta * p(domain) * np.log(q(domain) / p(domain) + epsilon)).sum()
