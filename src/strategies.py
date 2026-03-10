from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from src.interfaces import ClassifierStrategy, log_method

class DecisionTreeStrategy(ClassifierStrategy):
    def __init__(self, criterion='gini', max_depth=None, **kwargs):
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class RandomForestStrategy(ClassifierStrategy):
    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class LDAStrategy(ClassifierStrategy):
    def __init__(self, solver='svd', **kwargs):
        # LDA kennt random_state nicht
        kwargs.pop('random_state', None)

        self.model = LinearDiscriminantAnalysis(solver=solver, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class SVCRbfStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, gamma='scale', **kwargs):
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class SVCLinearStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, **kwargs):
        self.model = SVC(kernel='linear', C=C, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class SVCPoly3Strategy(ClassifierStrategy):
    def __init__(self, C=1.0, degree=3, **kwargs):
        self.model = SVC(kernel='poly', degree=degree, C=C, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class LinearSVCStrategy(ClassifierStrategy):
    def __init__(self, C=1.0, loss='squared_hinge', max_iter=1000, **kwargs):
        self.model = LinearSVC(C=C, loss=loss, max_iter=max_iter, **kwargs)

    @log_method
    def fit(self, X, y):
        self.model.fit(X, y)

    @log_method
    def predict(self, X):
        return self.model.predict(X)

class XGBoostStrategy(ClassifierStrategy):
    def __init__(self, n_estimators=100, learning_rate=0.1, **kwargs):
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.label_encoder = LabelEncoder()

    @log_method
    def fit(self, X, y):
        # XGBoost benötigt numerische Labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    @log_method
    def predict(self, X):
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)