from src.interfaces import ClassifierStrategy


class SklearnAdapter(ClassifierStrategy):
    """
    Adapter-Pattern (GoF):
    Passt ein beliebiges sklearn-kompatibles Modell
    (mit fit/predict Methoden) an das ClassifierStrategy-Interface an.
    So können auch Modelle genutzt werden, die keine eigene
    Strategy-Klasse besitzen.
    """

    def __init__(self, sklearn_model):
        self.model = sklearn_model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
