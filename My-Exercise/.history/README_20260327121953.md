# AI Engineering Lab - Breast Cancer Classification

Dieses Projekt zeigt einen sauberen End-to-End-Workflow fuer ein klassisches Machine-Learning-Problem:
Vorhersage von Brustkrebs-Klassen auf Basis des `sklearn` Breast-Cancer-Datensatzes.

Der Fokus liegt auf:
- nachvollziehbarer Datenanalyse,
- reproduzierbarem Training,
- klarer Trennung zwischen Training und Inferenz,
- professioneller Projektstruktur fuer Portfolio und CV.

## Projektstruktur

```text
My-Exercise/
|-- data/
|-- models/
|-- notebooks/
|   |-- 9_breast_cancer.py
|   |-- Pipeline.ipynb
|-- outputs/
|-- src/
|   |-- __init__.py
|   |-- train.py
|   |-- predict.py
|-- README.md
```

## Features

- Explorative Datenanalyse mit Visualisierungen
- Feature-Selektion basierend auf Korrelation
- Logistic-Regression-Modell als Baseline
- Persistentes Speichern des trainierten Modells
- Separates Inferenzskript fuer Vorhersagen

## Voraussetzungen

- Python 3.10+
- Virtuelle Umgebung (empfohlen)
- Installierte Pakete: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

## Schnellstart

1. Virtuelle Umgebung aktivieren.
2. Modell trainieren:

```bash
python -m src.train
```

3. Beispielvorhersagen ausgeben:

```bash
python -m src.predict --samples 5
```

## Training

Das Training-Skript:
- laedt den Datensatz,
- erzeugt einen reproduzierbaren Train/Test-Split,
- trainiert ein Logistic-Regression-Modell,
- speichert Modell und Metriken in `models/` und `outputs/`.

## Inferenz

Das Inferenz-Skript kann:
- Samples aus dem Testset vorhersagen,
- optional Daten aus einer CSV laden,
- Klassenlabels und Wahrscheinlichkeiten ausgeben.

## Ergebniskennzahlen

Typische Metriken:
- Accuracy
- Precision
- Recall
- F1-Score

Die tatsaechlichen Ergebnisse werden pro Lauf in `outputs/metrics.json` abgelegt.

## CV-Highlights

Dieses Repository demonstriert:
- sauberes ML-Projekt-Setup,
- klare Trennung von Analyse, Training und Inferenz,
- reproduzierbare Experimente,
- gut lesbaren, wartbaren Python-Code.
