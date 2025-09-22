# # main.py

# import pandas as pd
# import json
# from pathlib import Path
# import sys
# import os

# # --- Ajouter le dossier racine au PYTHONPATH pour trouver le package prepare_data ---
# sys.path.append(str(Path(__file__).parent.resolve()))

# # --- Import du module de nettoyage et exploration ---
# from prepare_data.data_cleaner import (
#     clean_dataset,
#     get_file_extensions,
#     check_images_consistency,
#     images_without_annotations,
#     annotations_without_images,
#     detect_abnormal_annotations
# )

# # --- Définir les chemins ---
# ANNOTATIONS_FILE = "data/_annotations.coco.json"
# IMAGES_FOLDER = "data/images/"
# OUTPUT_FILE = "data/annotations_clean.json"

# # --- Fonctions pour charger et sauvegarder COCO ---
# def load_coco_json(file_path: str):
#     """Charge un fichier JSON COCO et renvoie images_df et annotations_df."""
#     with open(file_path, "r") as f:
#         coco = json.load(f)
#     images_df = pd.DataFrame(coco["images"])
#     annotations_df = pd.DataFrame(coco["annotations"])
#     return coco, images_df, annotations_df

# def save_coco_json(coco: dict, images_df: pd.DataFrame, annotations_df: pd.DataFrame, output_path: str):
#     """Sauvegarde un nouveau fichier JSON COCO avec les données nettoyées."""
#     coco["images"] = images_df.to_dict(orient="records")
#     coco["annotations"] = annotations_df.to_dict(orient="records")
#     with open(output_path, "w") as f:
#         json.dump(coco, f, indent=2, default=int)
#     print(f"[INFO] Fichier nettoyé sauvegardé → {output_path}")

# # --- Main ---
# def main():
#     print("[START] Exploration du dataset...")

#     # Charger le JSON COCO
#     coco, images_df, annotations_df = load_coco_json(ANNOTATIONS_FILE)

#     # 1. Extensions présentes dans le dossier images
#     extensions = get_file_extensions(IMAGES_FOLDER)
#     print(f"Extensions trouvées dans {IMAGES_FOLDER}: {extensions}")

#     # 2. Cohérence images déclarées vs réelles
#     consistency = check_images_consistency(images_df, IMAGES_FOLDER)
#     print("Cohérence des images:")
#     for k, v in consistency.items():
#         print(f"  {k}: {v}")

#     # 3. Images sans annotations
#     imgs_no_ann = images_without_annotations(images_df, annotations_df, IMAGES_FOLDER)
#     print(f"Images sans annotations: {len(imgs_no_ann)}")
#     if len(imgs_no_ann) > 0:
#         print(imgs_no_ann[["id","file_name"]])

#     # 4. Annotations sans images
#     ann_no_img = annotations_without_images(annotations_df, images_df)
#     print(f"Annotations sans images: {len(ann_no_img)}")
#     if len(ann_no_img) > 0:
#         print(ann_no_img[["id","image_id"]])

#     # 5. Annotations anormales
#     abnormal_ann = detect_abnormal_annotations(annotations_df)
#     print(f"Annotations avec bbox anormales: {len(abnormal_ann)}")
#     if len(abnormal_ann) > 0:
#         print(abnormal_ann[["id","image_id","bbox"]])

#     print("[END] Exploration terminée\n")
    
#     # --- Nettoyage du dataset ---
#     print("[START] Nettoyage du dataset...")
#     images_df_clean, annotations_df_clean = clean_dataset(images_df, annotations_df, IMAGES_FOLDER)
#     save_coco_json(coco, images_df_clean, annotations_df_clean, OUTPUT_FILE)
#     print("[END] Nettoyage terminé")

# # --- Exécution ---
# if __name__ == "__main__":
#     main()


# main.py

from prepare_data.pipeline import run_pipeline

if __name__ == "__main__":
    # Fichiers et dossiers à adapter selon ton projet
    annotations_file = "data/_annotations.coco.json"   # chemin vers ton COCO JSON brut
    images_folder = "data/images"                # dossier contenant les images
    output_file = "data/annotations_clean.json"  # fichier de sortie nettoyé

    run_pipeline(annotations_file, images_folder, output_file)
