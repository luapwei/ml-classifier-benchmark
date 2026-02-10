from sklearn.tree import DecisionTreeClassifier
from src.interfaces import ClassifierStrategy

class DecisionTreeStrategy(ClassifierStrategy):
    """
    Konkrete Strategie für den Decision Tree Classifier.
    """
    def __init__(self, **kwargs):
        # Initialisiert das echte Sklearn-Modell mit den übergebenen Parametern
        # kwargs erlaubt uns z.B. max_depth oder random_state durchzureichen
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        # Delegiert den Aufruf an das Sklearn-Modell
        self.model.fit(X, y)

    def predict(self, X):
        # Liefert die Vorhersagen zurück
        return self.model.predict(X)