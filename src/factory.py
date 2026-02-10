from src.strategies import DecisionTreeStrategy


# Hier später die anderen 7 Strategien importieren

class ClassifierFactory:
    """
    Zentralisiert die Erzeugung der Strategie-Objekte.
    Eliminiert Fallunterscheidungen im Trainer-Code.
    """

    # Mapping von Bezeichnern zu Klassen (First-Class Objects)
    _strategies = {
        "tree": DecisionTreeStrategy,
        # "rf": RandomForestStrategy, usw.
    }

    @staticmethod
    def make_classifier(name: str, **kwargs):
        """
        Erzeugt eine Instanz der gewünschten Strategie.
        Gibt ein Objekt zurück, das dem ClassifierStrategy-Interface entspricht.
        """
        strategy_class = ClassifierFactory._strategies.get(name.lower())

        if not strategy_class:
            raise ValueError(f"Strategie '{name}' ist nicht im System hinterlegt.")

        # Instanziierung mit dynamischen Parametern
        return strategy_class(**kwargs)