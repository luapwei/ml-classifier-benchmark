# ML-Klassifikations-Benchmark – Projektablauf & Aktivitätsdiagramm

## Projektübersicht

Das Projekt vergleicht **8 ML-Klassifikationsmodelle** anhand des **Iris-Datensatzes** (150 Samples). Es nutzt die Design Patterns **Strategy** und **Factory**, um Modelle austauschbar und erweiterbar zu halten.

## Architektur (Klassenstruktur)

```mermaid
classDiagram
    direction TB
    class main["main.py"] {
        +main()
    }
    class Trainer {
        -train_data
        -test_data
        -features: list
        -target: str
        -results: dict
        +prepare_data()
        +run_benchmark(strategy_names, plot)
        -_print_summary()
    }
    class ClassifierFactory {
        -_strategies: dict
        +make_classifier(name, **kwargs)$
    }
    class ClassifierStrategy {
        <<abstract>>
        +fit(X, y)*
        +predict(X)*
    }
    class Metric {
        +calculate_error_rate(y_true, y_pred)$
        +plot_confusion_matrix(y_true, y_pred, name)$
    }

    main --> Trainer : erstellt
    Trainer --> ClassifierFactory : nutzt
    Trainer --> Metric : nutzt
    ClassifierFactory --> ClassifierStrategy : erzeugt
    ClassifierStrategy <|-- DecisionTreeStrategy
    ClassifierStrategy <|-- RandomForestStrategy
    ClassifierStrategy <|-- LDAStrategy
    ClassifierStrategy <|-- SVCRbfStrategy
    ClassifierStrategy <|-- SVCLinearStrategy
    ClassifierStrategy <|-- SVCPoly3Strategy
    ClassifierStrategy <|-- LinearSVCStrategy
    ClassifierStrategy <|-- XGBoostStrategy
```

## Detaillierter Ablauf (Schritt für Schritt)

### 1. Programmstart – `main.py`
- Logging wird konfiguriert (`INFO`-Level)
- Ein `Trainer`-Objekt wird erstellt (leere Datenfelder, Feature-/Target-Namen definiert)

### 2. Datenvorbereitung – `Trainer.prepare_data()`
- **Seed setzen**: `np.random.seed(123)` für Reproduzierbarkeit
- **Iris-Datensatz laden**: via `seaborn.load_dataset('iris')` (150 Samples, 4 Features, 3 Klassen)
- **Zufällige Permutation** der 150 Indizes
- **Split**: Die ersten **15 Indizes → Testdaten**, die restlichen **135 → Trainingsdaten**

### 3. Benchmark-Schleife – `Trainer.run_benchmark()`
Für **jedes der 8 Modelle** (`tree`, `rf`, `lda`, `svm_rbf`, `svm_linear`, `svm_poly3`, `linear_svc`, `xgboost`) wird folgender Zyklus durchlaufen:

| Schritt | Was passiert | Modul |
|---------|-------------|-------|
| **3a** | `ClassifierFactory.make_classifier(name)` – Sucht den Key im `_strategies`-Dict und instanziiert die konkrete Strategy-Klasse mit `random_state=123` | `factory.py` |
| **3b** | `strategy.fit(X_train, y_train)` – Trainiert das Modell auf den 135 Trainingsdaten | `strategies.py` |
| **3c** | `strategy.predict(X_test)` – Erzeugt Vorhersagen für die 15 Testdaten | `strategies.py` |
| **3d** | `Metric.calculate_error_rate(y_true, y_pred)` – Berechnet `1 - accuracy_score` | `metrics.py` |
| **3e** | `Metric.plot_confusion_matrix(...)` – Erstellt und speichert Confusion-Matrix-Heatmap als PNG in `results/` | `metrics.py` |
| **3f** | Error Rate wird in `self.results[name]` gespeichert | `trainer.py` |

### 4. Zusammenfassung – `Trainer._print_summary()`
- Sortiert alle 8 Modelle nach Error Rate (aufsteigend)
- Gibt ein Ranking mit Rang, Modellname, Error Rate und Accuracy aus

### Besonderheiten einzelner Strategien
- **LDA**: Entfernt `random_state` aus `kwargs`, da `LinearDiscriminantAnalysis` diesen Parameter nicht unterstützt
- **XGBoost**: Nutzt einen `LabelEncoder`, um String-Labels in numerische Werte zu konvertieren (und bei `predict` wieder zurück)
- **Alle Strategien**: Nutzen den `@log_method`-Decorator, der Klassenname, Methodenname und Anzahl der Samples loggt

---

## Aktivitätsdiagramm (mit Swimlanes)

Jede Swimlane zeigt die **zuständige Komponente** für den jeweiligen Schritt.

```mermaid
flowchart TD
    Start(["▶ Start"])

    subgraph MAIN["🟢 main.py"]
        direction TB
        M1["Logging konfigurieren<br/>(INFO-Level)"]
        M2["Trainer-Objekt erstellen"]
        M3["8 Strategie-Keys definieren:<br/>tree, rf, lda, svm_rbf,<br/>svm_linear, svm_poly3,<br/>linear_svc, xgboost"]
        M4["trainer.run_benchmark()<br/>aufrufen"]
    end

    subgraph TRAINER["🔵 Trainer"]
        direction TB
        T1["prepare_data():<br/>Seed(123), Iris laden,<br/>Indizes permutieren"]
        T2["Split: 15 Test / 135 Train"]
        LoopStart{"Nächste<br/>Strategie?"}
        T3["Error Rate in<br/>results-Dict speichern"]
        T4["Modellname und<br/>Error Rate ausgeben"]
        PlotCheck{"Visualisierung<br/>aktiviert?"}
        T5["_print_summary():<br/>Modelle nach Error Rate<br/>sortieren"]
        T6["Ranking-Tabelle ausgeben"]
    end

    subgraph FACTORY["🟣 ClassifierFactory"]
        direction TB
        F1["make_classifier(name):<br/>Key in _strategies<br/>nachschlagen"]
        F2["Konkrete Strategy-Klasse<br/>instanziieren<br/>(random_state=123)"]
    end

    subgraph STRATEGY["🟠 Strategy"]
        direction TB
        S1["fit(X_train, y_train)<br/>135 Samples trainieren"]
        S2["predict(X_test)<br/>15 Samples vorhersagen"]
    end

    subgraph METRIC["🔴 Metric"]
        direction TB
        ME1["calculate_error_rate()<br/>Error Rate = 1 − Accuracy"]
        ME2["plot_confusion_matrix()<br/>Heatmap als PNG<br/>in results/ speichern"]
    end

    Ende(["⏹ Ende"])

    Start --> M1 --> M2 --> T1 --> T2
    T2 --> M3 --> M4 --> LoopStart

    LoopStart -- Ja --> F1 --> F2
    F2 --> S1 --> S2
    S2 --> ME1 --> T3
    T3 --> PlotCheck
    PlotCheck -- Ja --> ME2 --> T4
    PlotCheck -- Nein --> T4
    T4 --> LoopStart

    LoopStart -- "Nein<br/>(alle 8 fertig)" --> T5 --> T6 --> Ende

    style Start fill:#2ecc71,color:#fff,stroke:#27ae60
    style Ende fill:#e74c3c,color:#fff,stroke:#c0392b
    style LoopStart fill:#f39c12,color:#fff,stroke:#e67e22
    style PlotCheck fill:#f39c12,color:#fff,stroke:#e67e22
    style MAIN fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20
    style TRAINER fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#0d47a1
    style FACTORY fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c
    style STRATEGY fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
    style METRIC fill:#fce4ec,stroke:#f44336,stroke-width:2px,color:#b71c1c
```

---

## Sequenzdiagramm

Zeigt die **zeitliche Abfolge der Aufrufe** zwischen den Komponenten.

```mermaid
sequenceDiagram
    actor User
    participant M as main.py
    participant T as Trainer
    participant F as ClassifierFactory
    participant S as Strategy
    participant ME as Metric

    User->>M: python main.py
    activate M
    M->>M: Logging konfigurieren (INFO)
    M->>T: Trainer() erstellen
    activate T
    M->>T: prepare_data()
    T->>T: np.random.seed(123)
    T->>T: sns.load_dataset("iris") → 150 Samples
    T->>T: Indizes permutieren
    T->>T: Split: 15 Test / 135 Train
    T-->>M: Daten bereit

    M->>T: run_benchmark(strategies, visualisierung=True)

    loop Für jede der 8 Strategien
        T->>F: make_classifier(name, random_state=123)
        activate F
        F->>F: Key in _strategies nachschlagen
        F->>S: Strategy-Klasse instanziieren
        activate S
        F-->>T: Strategy-Objekt zurückgeben
        deactivate F

        T->>S: fit(X_train, y_train) — 135 Samples
        S->>S: Modell trainieren
        S-->>T: Training abgeschlossen

        T->>S: predict(X_test) — 15 Samples
        S->>S: Vorhersagen berechnen
        S-->>T: y_pred zurückgeben
        deactivate S

        T->>ME: calculate_error_rate(y_true, y_pred)
        activate ME
        ME->>ME: accuracy_score berechnen
        ME-->>T: Error Rate = 1 − Accuracy
        deactivate ME

        T->>T: Error Rate in results speichern

        alt Visualisierung aktiviert
            T->>ME: plot_confusion_matrix(y_true, y_pred, name)
            activate ME
            ME->>ME: Heatmap erstellen
            ME->>ME: PNG in results/ speichern
            ME-->>T: Grafik gespeichert
            deactivate ME
        end

        T->>T: Modellname & Error Rate ausgeben
    end

    T->>T: _print_summary()
    T->>T: Modelle nach Error Rate sortieren
    T->>T: Ranking-Tabelle ausgeben
    T-->>M: Benchmark abgeschlossen
    deactivate T
    M-->>User: Programm beendet
    deactivate M
```

---

## Dateiübersicht

| Datei | Rolle | Zeilen |
|-------|-------|--------|
| [main.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/main.py) | Einstiegspunkt, definiert Strategie-Liste, startet Benchmark | 42 |
| [trainer.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/src/trainer.py) | Orchestrierung: Daten laden, split, Schleife über Modelle, Zusammenfassung | 63 |
| [factory.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/src/factory.py) | Factory Pattern: String-Key → konkretes Strategy-Objekt | 45 |
| [interfaces.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/src/interfaces.py) | Abstrakte Basisklasse `ClassifierStrategy` + `@log_method` Decorator | 26 |
| [strategies.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/src/strategies.py) | 8 konkrete Strategy-Implementierungen | 117 |
| [metrics.py](file:///c:/Code/AUGSBURG_10237747_Python_Simon/src/metrics.py) | Error Rate berechnen + Confusion-Matrix-Heatmap speichern | 29 |
