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
    """
    Die Factory instanziiert die konkreten Strategien basierend auf einem String-Key.
    Dies verhindert manuelle Fallunterscheidungen im Trainer.
    """

    # Das Mapping-Dictionary gemäß deiner Mapping-Tabelle
    _strategies = {
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
    def make_classifier(name, **kwargs):
        """
        Erzeugt das Strategie-Objekt.
        :param name: Der Key aus dem _strategies Dictionary.
        :param kwargs: Optionale Parameter für das Modell (z.B. max_depth).
        :return: Eine Instanz einer ClassifierStrategy.
        """
        strategy_class = ClassifierFactory._strategies.get(name.lower())

        if not strategy_class:
            raise ValueError(f"Strategie '{name}' ist nicht bekannt. "
                             f"Verfügbar sind: {list(ClassifierFactory._strategies.keys())}")

        # Instanziierung der Klasse mit den übergebenen Parametern
        return strategy_class(**kwargs)