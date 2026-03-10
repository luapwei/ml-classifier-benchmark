# ML-Klassifikations-Benchmark
### README.md wurde durch Claude (claude.ai) erstellt

Dieses Projekt vergleicht 8 verschiedene Machine-Learning-Klassifikationsmodelle anhand des Iris-Datensatzes. Es wurde im Rahmen einer Studienarbeit an der Universität Augsburg entwickelt.

## Modelle

| Nr. | Modell | Key |
|-----|--------|-----|
| 1 | Decision Tree | `tree` |
| 2 | Random Forest | `rf` |
| 3 | Linear Discriminant Analysis | `lda` |
| 4 | SVM (RBF-Kernel) | `svm_rbf` |
| 5 | SVM (Linear-Kernel) | `svm_linear` |
| 6 | SVM (Polynomial, Grad 3) | `svm_poly3` |
| 7 | LinearSVC | `linear_svc` |
| 8 | XGBoost | `xgboost` |

## Architektur

Das Projekt nutzt die Design Patterns **Strategy** und **Factory**:

```
main.py                  → Einstiegspunkt
  └─ Trainer             → Datenvorbereitung & Benchmark-Orchestrierung
      ├─ ClassifierFactory   → Erzeugt Modelle über String-Key
      │   └─ ClassifierStrategy (ABC) → 8 konkrete Strategien
      └─ Metric              → Error Rate & Confusion Matrix
```

- **`src/interfaces.py`** – Abstrakte Basisklasse `ClassifierStrategy`
- **`src/strategies.py`** – Konkrete Modell-Strategien
- **`src/factory.py`** – Factory zur Modell-Erzeugung
- **`src/metrics.py`** – Fehlerrate und Confusion-Matrix-Plots
- **`src/trainer.py`** – Datenaufbereitung und Benchmark-Ablauf

## Installation

```bash
# Repository klonen
git clone <repository-url>
cd AUGSBURG_10237747_Python_Simon

# Virtual Environment erstellen und aktivieren
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Ausführung

```bash
python main.py
```

Das Programm:
1. Lädt den Iris-Datensatz (150 Samples)
2. Splittet in 15 Test- und 135 Trainingsdaten
3. Trainiert und evaluiert alle 8 Modelle
4. Speichert Confusion-Matrix-Heatmaps im Ordner `results/`
5. Gibt eine sortierte Ergebnis-Zusammenfassung mit Error Rates aus

## Ergebnisse

Die generierten Confusion-Matrix-Heatmaps werden als PNG-Dateien im Ordner `results/` gespeichert. Nach dem Durchlauf wird ein Ranking aller Modelle nach Error Rate ausgegeben.
