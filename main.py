import logging
from src.trainer import Trainer
from src.factory import ClassifierFactory
from src.adapter import SklearnAdapter

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    """
    Client (GoF Strategy Pattern):
    Wählt die konkreten Strategien über die Factory
    und injiziert sie in den Kontext (Trainer).

    Demonstriert zusätzlich:
    - Adapter-Pattern   (SklearnAdapter)
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


    # 5. Zusammenfassung ausgeben
    trainer.print_summary()


if __name__ == "__main__":
    main()

