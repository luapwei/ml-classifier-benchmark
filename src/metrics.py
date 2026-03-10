from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Metric:
    @staticmethod
    def calculate_error_rate(y_true, y_pred):
        # Berechnet Fehlerrate
        accuracy = accuracy_score(y_true, y_pred)
        return 1 - accuracy

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name):
        # Erstellt Ordner results
        pfad = "results"
        if not os.path.exists(pfad):
            os.makedirs(pfad)

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')

        # Speichert die Heatmap
        dateiname = f"{pfad}/heatmap_{model_name}.png"
        plt.savefig(dateiname)
        print(f"Grafik gespeichert: {dateiname}")

        plt.close() # Ohne das läuft die Loop nicht weiter