🔥 Détection d’incendies sur images satellites

Un projet de vision par ordinateur utilisant YOLOv8/YOLOv9 pour détecter et localiser les zones brûlées ou en feu à partir d’images satellites.
L’objectif est de contribuer à la prévention et au suivi des incendies afin d’aider les pompiers et services de gestion des risques.

🚀 Fonctionnalités principales

Prétraitement des données et conversion au format YOLO.

Entraînement d’un modèle YOLO (v8 ou v9) sur les données satellites.

Suivi des métriques de performance (Precision, Recall, mAP).

Visualisation des prédictions directement sur les images.

Sauvegarde des modèles entraînés et reprise des expériences.

📂 Structure du projet
detection-incendies/
├── data/                  # Données brutes et préparées
│   ├── raw/               # Données originales (COCO, etc.)
│   └── dataset_yolo/      # Données converties au format YOLO
├── src/
│   ├── data_preparation_yolo.py   # Script de préparation des données
│   ├── train.py                   # Script d’entraînement
├── notebooks/
│   └── modele_interpretation.ipynb  # Notebook Colab pour expérimenter
├── requirements.txt
└── README.md

⚙️ Installation
1. Cloner le dépôt
git clone https://github.com/Thibaud34/detection-incendies
cd detection-incendies

2. Installer les dépendances
pip install -r requirements.txt

📊 Pipeline d’entraînement
1. Préparation des données

Convertir vos données au format YOLO
Ce travail ne propose pas de data, vous devez exporter votre propre jeu de données.



2. Entraînement d’un modèle YOLO

Exemple avec YOLOv9 Small :

Utilisation de Google Collab


Les résultats (métriques, logs, modèles) seront sauvegardés dans runs/detect/.

🔍 Inférence (prédictions sur de nouvelles images)

Les images annotées avec les prédictions seront disponibles dans runs/detect/predict/.

🎯 Objectif final

Faciliter la détection automatique des zones affectées par les incendies pour améliorer la réactivité et la planification des interventions.