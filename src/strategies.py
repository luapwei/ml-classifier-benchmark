from __future__ import annotations

from typing import Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from src.interfaces import ClassifierStrategy, log_method

class DecisionTreeStrategy(ClassifierStrategy):
    def __init__(self, criterion: str = 'gini', max_depth: int | None = None, **kwargs: Any) -> None:
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class RandomForestStrategy(ClassifierStrategy):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class LDAStrategy(ClassifierStrategy):
    def __init__(self, solver: str = 'svd', **kwargs: Any) -> None:
        # LDA kennt random_state nicht
        kwargs.pop('random_state', None)

        self.model = LinearDiscriminantAnalysis(solver=solver, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class SVCRbfStrategy(ClassifierStrategy):
    def __init__(self, C: float = 1.0, gamma: str = 'scale', **kwargs: Any) -> None:
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class SVCLinearStrategy(ClassifierStrategy):
    def __init__(self, C: float = 1.0, **kwargs: Any) -> None:
        self.model = SVC(kernel='linear', C=C, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class SVCPoly3Strategy(ClassifierStrategy):
    def __init__(self, C: float = 1.0, degree: int = 3, **kwargs: Any) -> None:
        self.model = SVC(kernel='poly', degree=degree, C=C, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class LinearSVCStrategy(ClassifierStrategy):
    def __init__(self, C: float = 1.0, loss: str = 'squared_hinge', max_iter: int = 1000, **kwargs: Any) -> None:
        self.model = LinearSVC(C=C, loss=loss, max_iter=max_iter, **kwargs)

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class XGBoostStrategy(ClassifierStrategy):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, **kwargs: Any) -> None:
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.label_encoder = LabelEncoder()

    @log_method
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # XGBoost benötigt numerische Labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    @log_method
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds: np.ndarray = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)