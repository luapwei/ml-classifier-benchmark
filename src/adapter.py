from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.interfaces import ClassifierStrategy, log_method


class SklearnAdapter(ClassifierStrategy):
    # Adapter-Pattern passt ein beliebiges sklearn-Modell wie das ClassifierStrategy interface an
    def __init__(self, sklearn_model: BaseEstimator) -> None:
        self.model = sklearn_model

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
