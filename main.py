import logging
from src.trainer import Trainer
from src.factory import ClassifierFactory

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    """
    Client (GoF Strategy Pattern):
    Wählt die konkreten Strategien über die Factory
    und injiziert sie in den Kontext (Trainer).
    """

    # 1. Kontext erstellen
    trainer = Trainer()

    # 2. Daten laden und splitten
    trainer.prepare_data()

    # 3. Liste der 8 Strategien
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

    # 5. Zusammenfassung ausgeben
    trainer.print_summary()


if __name__ == "__main__":
    main()
