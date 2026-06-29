from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


class Metric:
    @staticmethod
    def calculate_error_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
        # Berechnet Fehlerrate
        accuracy: float = accuracy_score(y_true, y_pred)
        return 1 - accuracy

    @staticmethod
    def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
        # Erstellt Ordner results
        pfad: str = "results"
        if not os.path.exists(pfad):
            os.makedirs(pfad)

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')

        # Speichert die Heatmap
        dateiname: str = f"{pfad}/heatmap_{model_name}.png"
        plt.savefig(dateiname)
        print(f"Grafik gespeichert: {dateiname}")

        plt.close() # Ohne das läuft die Loop nicht weiter