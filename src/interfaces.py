from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import functools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_method(func: Callable[..., Any]) -> Callable[..., Any]:
    # Decorator-Pattern für Logging
    @functools.wraps(func)
    def wrapper(self: ClassifierStrategy, *args: Any, **kwargs: Any) -> Any:
        logger.info(f"{self.__class__.__name__}.{func.__name__}() – {len(args[0])} Samples")
        return func(self, *args, **kwargs)
    return wrapper


class ClassifierStrategy(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass