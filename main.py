import logging
import sys
from src.trainer import Trainer
from src.factory import ClassifierFactory
from src.adapter import SklearnAdapter

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout
)

def main():

    # Kontext erstellen
    trainer = Trainer()

    # Daten laden und splitten
    trainer.prepare_data()

    # Liste der Strategien
    strategy_names = [
        "tree",
        "rf",
        "lda",
        "svm_rbf",
        "svm_linear",
        "svm_poly3",
        "linear_svc",
        "xgboost"
    ]

    # Heatmap aktiviert
    visualisierung = True

    # Benchmark
    try:
        for name in strategy_names:
            # Client erstellt die Strategie via Factory
            strategy = ClassifierFactory.make_classifier(name, random_state=123)
            # Client injiziert sie in den Kontext
            trainer.run_single(strategy, name, visualisierung)
    except Exception as e:
        print(f"Fehler: {e}")

    # Adapter-Pattern adaptiert ein beliebiges sklearn-Modell ohne Klasse in strategies.py
    print("\nAdapter-Pattern: KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    knn_adapter = SklearnAdapter(KNeighborsClassifier(n_neighbors=3))
    trainer.run_single(knn_adapter, "adapter_knn", visualisierung)

    # Zusammenfassung
    trainer.print_summary()

if __name__ == "__main__":
    main()

