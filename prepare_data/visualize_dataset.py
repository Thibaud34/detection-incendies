# visualize_dataset.py

import fiftyone as fo
import fiftyone.types as fot

# --- Chemins ---
IMAGES_DIR = "data/images/"                 # dossier des images
COCO_JSON_PATH = "data/annotations_clean.json"  # fichier COCO JSON nettoyé

# --- Charger le dataset COCO via Dataset.from_dir() ---
dataset = fo.Dataset.from_dir(
    dataset_type=fot.COCODetectionDataset,
    data_path=IMAGES_DIR,
    labels_path=COCO_JSON_PATH,
    name="IncendiesClean",  # nom du dataset dans FiftyOne
    overwrite=True           # réécrit le dataset si déjà présent
)

# --- Lancer l'interface graphique pour visualiser ---
session = fo.launch_app(dataset, address="0.0.0.0", port=5151)
session.wait()
# --- Optionnel : résumé console ---
print(dataset)
