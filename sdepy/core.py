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


