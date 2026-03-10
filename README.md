# ML-Klassifikations-Benchmark
### README.md wurde durch Claude (claude.ai) erstellt

Dieses Projekt vergleicht verschiedene Machine-Learning-Klassifikationsmodelle anhand des Iris-Datensatzes. Es wurde im Rahmen einer Studienarbeit an der Universität Augsburg entwickelt.

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

Zusätzlich wird über das **Adapter-Pattern** ein `KNeighborsClassifier` als 9. Modell eingebunden, ohne dass dafür eine eigene Strategy-Klasse benötigt wird.

## GoF Design Patterns

Das Projekt setzt vier Gang-of-Four (GoF) Design Patterns ein:

### 1. Strategy Pattern
Die abstrakte Klasse `ClassifierStrategy` definiert eine einheitliche Schnittstelle (`fit`, `predict`). Jedes der 8 ML-Modelle ist als eigene konkrete Strategie implementiert. Der `Trainer` arbeitet nur gegen die abstrakte Schnittstelle und ist dadurch vom konkreten Modell entkoppelt.

- **Interface:** `ClassifierStrategy` in `src/interfaces.py`
- **Konkrete Strategien:** `DecisionTreeStrategy`, `RandomForestStrategy`, `LDAStrategy`, `SVCRbfStrategy`, `SVCLinearStrategy`, `SVCPoly3Strategy`, `LinearSVCStrategy`, `XGBoostStrategy` in `src/strategies.py`
- **Kontext:** `Trainer.run_single()` in `src/trainer.py`

### 2. Factory Pattern
Die `ClassifierFactory` erzeugt über einen String-Key die passende Strategy-Instanz. Der Client (`main.py`) muss die konkreten Klassen nicht kennen.

- **Factory:** `ClassifierFactory.make_classifier()` in `src/factory.py`

### 3. Decorator Pattern
Die Funktion `log_method` ist ein Python-Decorator, der das Logging transparent um die `fit`- und `predict`-Methoden der Strategien wickelt — ohne deren Code zu verändern. Dies entspricht dem GoF-Decorator-Pattern: zusätzliche Verantwortlichkeit (Logging) wird dynamisch hinzugefügt.

- **Decorator:** `@log_method` in `src/interfaces.py`
- **Anwendung:** Auf `fit()` und `predict()` aller konkreten Strategien in `src/strategies.py`

### 4. Adapter Pattern
Der `SklearnAdapter` passt ein beliebiges sklearn-Modell an die `ClassifierStrategy`-Schnittstelle an. So können Modelle ohne eigene Strategy-Klasse direkt im Benchmark verwendet werden (z. B. `KNeighborsClassifier`).

- **Adapter:** `SklearnAdapter` in `src/adapter.py`
- **Anwendung:** In `main.py` wird `KNeighborsClassifier` über den Adapter eingebunden

## Architektur

```
main.py                    → Einstiegspunkt
  └─ Trainer               → Datenvorbereitung & Benchmark-Orchestrierung
      ├─ ClassifierFactory  → Erzeugt Modelle über String-Key (Factory)
      │   └─ ClassifierStrategy (ABC)
      │       └─ 8 konkrete Strategien (Strategy)
      ├─ SklearnAdapter     → Passt beliebige sklearn-Modelle an (Adapter)
      ├─ @log_method        → Logging-Decorator für fit/predict (Decorator)
      └─ Metric             → Error Rate & Confusion Matrix
```

### Projektstruktur

| Datei | Beschreibung |
|-------|-------------|
| `main.py` | Einstiegspunkt: Benchmark-Ablauf und Adapter-Demo |
| `src/interfaces.py` | Abstrakte Basisklasse `ClassifierStrategy` + `@log_method` Decorator |
| `src/strategies.py` | 8 konkrete Modell-Strategien |
| `src/factory.py` | `ClassifierFactory` zur Modell-Erzeugung per String-Key |
| `src/adapter.py` | `SklearnAdapter` für beliebige sklearn-Modelle |
| `src/metrics.py` | Fehlerrate-Berechnung und Confusion-Matrix-Heatmaps |
| `src/trainer.py` | Datenaufbereitung (Iris-Datensatz) und Benchmark-Ablauf |

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
3. Trainiert und evaluiert alle 8 Strategy-Modelle (via Factory)
4. Bindet ein zusätzliches Modell über den Adapter ein
5. Speichert Confusion-Matrix-Heatmaps im Ordner `results/`
6. Gibt eine sortierte Ergebnis-Zusammenfassung mit Error Rates aus
