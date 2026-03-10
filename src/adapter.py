from src.interfaces import ClassifierStrategy


class SklearnAdapter(ClassifierStrategy):
    # Adapter-Pattern passt ein beliebiges sklearn-Modell wie das ClassifierStrategy interface an
    def __init__(self, sklearn_model):
        self.model = sklearn_model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
