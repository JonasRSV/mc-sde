from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class SDE(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self) -> Tuple[np.float64, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def steps(self) -> int:  # returns the total number of steps left
        raise NotImplementedError()


class PDF(ABC):

    @abstractmethod
    def merge(self, pdf: "PDF"):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, particles: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, particles: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def mean(self) -> float:  # mean of distribution
        raise NotImplementedError()


class Job:
    RAW = 0
    Video = 1

    def __init__(self, sde: SDE, pdf: PDF, mode: int, settings: dict):
        self.sde = sde
        self.pdf = pdf
        self.mode = mode
        self.settings = settings

    def init_on_process(self):
        self.sde.preprocess()
        self.pdf.preprocess()


class Result:
    def __init__(self, distributions):
        self.distributions: List[PDF] = distributions

    def extend(self, result):
        self.distributions.extend(result.distributions)
