from abc import ABC, abstractmethod
from typing import Tuple
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


class Job:
    RAW = 0

    def __init__(self, sde: SDE, pdf: PDF, mode: int):
        self.sde = sde
        self.pdf = pdf
        self.mode = mode

    def init_on_process(self):
        self.sde.preprocess()
        self.pdf.preprocess()
