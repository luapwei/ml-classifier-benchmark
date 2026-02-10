from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from src.interfaces import ClassifierStrategy

class DecisionTreeStrategy(ClassifierStrategy):
    def __init__(self, criterion='gini', max_depth=None, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestStrategy(ClassifierStrategy):
    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class LDAStrategy(ClassifierStrategy):
    def __init__(self, solver='svd', **kwargs):
        # Wir entfernen random_state, falls es übergeben wurde,
        # da LDA diesen Parameter nicht kennt.
        kwargs.pop('random_state', None)

        self.model = LinearDiscriminantAnalysis(solver=solver, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVCRbfStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, gamma='scale', **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVCLinearStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = SVC(kernel='linear', C=C, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVCPoly3Strategy(ClassifierStrategy):
    def __init__(self, C=1.0, degree=3, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = SVC(kernel='poly', degree=degree, C=C, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class LinearSVCStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, loss='squared_hinge', max_iter=1000, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = LinearSVC(C=C, loss=loss, max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class XGBoostStrategy(ClassifierStrategy):
    def __init__(self, n_estimators=100, learning_rate=0.1, **kwargs):
        # Parameter aus Mapping-Tabelle
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        # XGBoost benötigt numerische Labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    def predict(self, X):
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)