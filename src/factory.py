from __future__ import annotations

from typing import Any

from src.interfaces import ClassifierStrategy
from src.strategies import (
    DecisionTreeStrategy,
    RandomForestStrategy,
    LDAStrategy,
    SVCRbfStrategy,
    SVCLinearStrategy,
    SVCPoly3Strategy,
    LinearSVCStrategy,
    XGBoostStrategy
)


class ClassifierFactory:
    # Mapping-Dictionary > Verwendet in main
    _strategies: dict[str, type[ClassifierStrategy]] = {
        "tree": DecisionTreeStrategy,
        "rf": RandomForestStrategy,
        "lda": LDAStrategy,
        "svm_rbf": SVCRbfStrategy,
        "svm_linear": SVCLinearStrategy,
        "svm_poly3": SVCPoly3Strategy,
        "linear_svc": LinearSVCStrategy,
        "xgboost": XGBoostStrategy
    }

    @staticmethod
    def make_classifier(name: str, **kwargs: Any) -> ClassifierStrategy:
        # Strategie muss kleingeschrieben sein
        strategy_class = ClassifierFactory._strategies.get(name.lower())

        if not strategy_class:
            raise ValueError(f"Strategie '{name}' nicht bekannt")

        # Instanziierung der Klasse
        return strategy_class(**kwargs)