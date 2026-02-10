import numpy as np
import seaborn as sns
from src.factory import ClassifierFactory
from src.metrics import Metric


class Trainer:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target = 'species'
        self.results = {}

    def prepare_data(self):
        # Deine Logik: Iris laden und über Indizes permutieren
        np.random.seed(123)
        iris = sns.load_dataset('iris')
        indices = np.random.permutation(len(iris))

        # Aufteilung 15 (Test) zu Rest (Train) via iloc
        self.test_data = iris.iloc[indices[:15]]
        self.train_data = iris.iloc[indices[15:]]

    def run_benchmark(self, strategy_names):
        for name in strategy_names:
            # 1. Factory baut das Modell
            strategy = ClassifierFactory.make_classifier(name, random_state=123)

            # 2. Training mit deinen Feature/Target Definitionen
            strategy.fit(self.train_data[self.features], self.train_data[self.target])

            # 3. Vorhersage
            y_pred = strategy.predict(self.test_data[self.features])

            # 4. Auswertung
            y_true = self.test_data[self.target]
            # 1. Berechnung der Error Rate über die Metric-Klasse
            error_rate = Metric.calculate_error_rate(y_true, y_pred)
            self.results[name] = error_rate

            # 2. Erstellung der Heatmap über die Metric-Klasse
            Metric.plot_confusion_matrix(y_true, y_pred, name)
            print(f"{name} abgeschlossen. Error Rate: {error_rate:.4f}")