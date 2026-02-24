from src.trainer import Trainer


def main(): 
    # 1. Trainer erstellen
    trainer = Trainer()

    # 2. Daten laden und splitten (Deine 15-Test-Logik)
    trainer.prepare_data()

    # 3. Liste der 8 Strategien aus der PDF
    strategies = [
        "tree",
        "rf",
        "lda",
        "svm_rbf",
        "svm_linear",
        "svm_poly3",
        "linear_svc",
        "xgboost"
    ]

    print("--- Starte Mini-Keras Benchmark ---")

    # 4. Automatischer Durchlauf
    # Wir übergeben random_state, damit die Modelle (z.B. RF)
    # immer gleich trainieren.
    try:
        trainer.run_benchmark(strategies)
    except Exception as e:
        print(f"Fehler im Ablauf: {e}")


if __name__ == "__main__":
    main()