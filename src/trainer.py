import numpy as np
import seaborn as sns
from src.metrics import Metric


class Trainer:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target = 'species'
        self.results = {}

    def prepare_data(self):
        # Iris laden und über Indizes permutieren
        np.random.seed(123)
        iris = sns.load_dataset('iris')
        indices = np.random.permutation(len(iris))

        # Aufteilung 15 (Test) zu Rest (Train) via iloc
        self.test_data = iris.iloc[indices[:15]]
        self.train_data = iris.iloc[indices[15:]]

    def run_single(self, strategy, name, plot=True):

        # Training mit Feature/Target
        strategy.fit(self.train_data[self.features], self.train_data[self.target])

        # Vorhersage
        y_pred = strategy.predict(self.test_data[self.features])

        # Auswertung
        y_true = self.test_data[self.target]

        error_rate = Metric.calculate_error_rate(y_true, y_pred)
        self.results[name] = error_rate

        # Erstellung der Heatmap (optional)
        if plot:
            Metric.plot_confusion_matrix(y_true, y_pred, name)
        print(f"{name} abgeschlossen. Error Rate: {error_rate:.4f}")

    def print_summary(self):
        # Zusammenfassung: Ranking aller Modelle nach Error Rate
        print("\n" + "=" * 50)
        print("          Zusammenfassung")
        print("=" * 50)
        print(f"{'Rang':<6}{'Modell':<16}{'Error Rate':<12}{'Accuracy'}")

        # Ergebnisse sortieren - lambda x[1] sortiert nach Value
        sorted_results = sorted(self.results.items(), key=lambda x: x[1])
        for rang, (name, error_rate) in enumerate(sorted_results, start=1):
            accuracy = 1 - error_rate
            print(f"{rang:<6}{name:<16}{error_rate:<12.4f}{accuracy:.2%}")

        print("=" * 50)