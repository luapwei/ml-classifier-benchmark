import numpy as np
import seaborn as sns
from src.metrics import Metric


class Trainer:
    """
    Kontext (GoF Strategy Pattern).
    Der Trainer kennt nur das ClassifierStrategy-Interface,
    NICHT die Factory oder konkrete Strategien.
    """

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
        """
        Führt Training, Vorhersage und Auswertung
        mit einer vom Client injizierten Strategie durch.

        :param strategy: Eine Instanz von ClassifierStrategy (injiziert vom Client).
        :param name: Bezeichnung der Strategie für die Ergebnisausgabe.
        :param plot: Ob eine Confusion-Matrix-Heatmap erzeugt werden soll.
        """
        # 1. Training mit Feature/Target Definitionen
        strategy.fit(self.train_data[self.features], self.train_data[self.target])

        # 2. Vorhersage
        y_pred = strategy.predict(self.test_data[self.features])

        # 3. Auswertung
        y_true = self.test_data[self.target]

        error_rate = Metric.calculate_error_rate(y_true, y_pred)
        self.results[name] = error_rate

        # 4. Erstellung der Heatmap (optional)
        if plot:
            Metric.plot_confusion_matrix(y_true, y_pred, name)
        print(f"{name} abgeschlossen. Error Rate: {error_rate:.4f}")

    def print_summary(self):
        """Zusammenfassung: Ranking aller Modelle nach Error Rate."""
        print("\n" + "=" * 50)
        print("          ERGEBNIS-ZUSAMMENFASSUNG")
        print("=" * 50)
        print(f"{'Rang':<6}{'Modell':<16}{'Error Rate':<12}{'Accuracy'}")
        print("-" * 50)

        sorted_results = sorted(self.results.items(), key=lambda x: x[1])
        for rang, (name, error_rate) in enumerate(sorted_results, start=1):
            accuracy = 1 - error_rate
            print(f"{rang:<6}{name:<16}{error_rate:<12.4f}{accuracy:.2%}")

        print("=" * 50)