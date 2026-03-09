import logging
from src.trainer import Trainer
from src.factory import ClassifierFactory
from src.adapter import SklearnAdapter
from src.composite import EnsembleStrategy

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    """
    Client (GoF Strategy Pattern):
    Wählt die konkreten Strategien über die Factory
    und injiziert sie in den Kontext (Trainer).

    Demonstriert zusätzlich:
    - Adapter-Pattern   (SklearnAdapter)
    - Composite-Pattern (EnsembleStrategy)
    """

    # 1. Kontext erstellen
    trainer = Trainer()

    # 2. Daten laden und splitten
    trainer.prepare_data()

    # 3. Liste der 8 Strategien (Strategy + Factory Pattern)
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

    # Visualisierung (Heatmaps) ein/aus
    visualisierung = True

    print("--- Starte Mini-Keras Benchmark ---")

    # 4. Client wählt jede Strategie über die Factory
    #    und injiziert sie in den Kontext (Trainer)
    try:
        for name in strategy_names:
            # Client erstellt die Strategie via Factory
            strategy = ClassifierFactory.make_classifier(name, random_state=123)
            # Client injiziert sie in den Kontext
            trainer.run_single(strategy, name, visualisierung)
    except Exception as e:
        print(f"Fehler im Ablauf: {e}")

    # ── Adapter-Pattern ─────────────────────────────────────────
    # Adaptiert ein beliebiges sklearn-Modell auf das
    # ClassifierStrategy-Interface, ohne eine eigene Strategy-Klasse.
    print("\n--- Adapter-Pattern: KNeighborsClassifier ---")
    from sklearn.neighbors import KNeighborsClassifier
    knn_adapter = SklearnAdapter(KNeighborsClassifier(n_neighbors=3))
    trainer.run_single(knn_adapter, "adapter_knn", visualisierung)

    # ── Composite-Pattern ───────────────────────────────────────
    # Bündelt mehrere Strategien zu einem Ensemble,
    # Vorhersage per Majority-Voting.
    print("\n--- Composite-Pattern: Ensemble (RF + SVM + XGB) ---")
    ensemble = EnsembleStrategy()
    ensemble.add(ClassifierFactory.make_classifier("rf", random_state=123))
    ensemble.add(ClassifierFactory.make_classifier("svm_rbf", random_state=123))
    ensemble.add(ClassifierFactory.make_classifier("xgboost", random_state=123))
    trainer.run_single(ensemble, "composite_ensemble", visualisierung)

    # 5. Zusammenfassung ausgeben
    trainer.print_summary()


if __name__ == "__main__":
    main()

