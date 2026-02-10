from abc import ABC, abstractmethod

class ClassifierStrategy(ABC):
    """
    Abstrakte Basisklasse für alle Klassifikations-Strategien.
    Stellt sicher, dass jede Strategie fit() und predict() implementiert.
    """

    @abstractmethod
    def fit(self, x, y):
        """Trainiert das Modell mit Features X und Labels y."""
        pass

    @abstractmethod
    def predict(self, x):
        """Liefert Vorhersagen für die gegebenen Features X."""
        pass