# Application de Détection d’Encombrants par IA

Projet L3 Informatique - Aix-Marseille Université (2025-2026)

Système de détection automatique d’encombrants urbains par caméra, combinant soustraction de fond statique (OpenCV) et classification par modèle YOLOv8m entraîné sur mesure (11 classes, 89.1% mAP50). Les signalements sont enregistrés en base SQLite et consultables via une interface web Flask avec carte interactive.

## Stack technique

- **Python 3.14** - Langage principal
- **OpenCV** - Capture vidéo, soustraction de fond, morphologie
- **YOLOv8m (Ultralytics)** - Classification des encombrants
- **Flask** - Serveur web
- **SQLite** - Base de données
- **Leaflet.js + OpenStreetMap** - Carte interactive

## Structure du projet

```
detector.py          # Module de détection (fond statique + persistance + YOLO)
database.py          # Accès centralisé à la base de données
predict_img.py       # Classification sur image fixe
split_dataset.py     # Découpage du dataset d’entraînement
bdd/                 # Base de données SQLite (générée automatiquement)
runs/detect/train-19/weights/best.pt  # Poids du modèle YOLO
siteweb/
  app.py             # Serveur Flask
  templates/         # Pages HTML (accueil, détail, caméras, historique, modèle, carte)
  static/            # Captures, icônes
```

## Installation

```bash
pip install opencv-python ultralytics flask numpy
```

## Lancement

Terminal 1 (détecteur) :
```bash
python3 detector.py
```

Terminal 2 (serveur web) :
```bash
cd siteweb
python3 app.py
```

Interface accessible sur http://localhost:5000

## Classes détectées (11)

Canapé, Carton, Chaise, Commode, Réfrigérateur, Machine à laver, Matelas, Sac poubelle, Table, Vélo, Télévision

## Auteur

Balme Maxence - L3 Informatique, Aix-Marseille Université
