# pipeline.py

from prepare_data.data_loader import load_coco_annotations, coco_to_dataframes, save_coco_annotations
from prepare_data.data_explorer import explore_dataset
from prepare_data.data_cleaner import (
    clean_dataset,
    annotations_without_images
)


def run_pipeline(annotations_file: str, images_folder: str, output_file: str):
    """Pipeline complet d'exploration et de nettoyage COCO."""

    # --- 1. Charger les données ---
    coco = load_coco_annotations(annotations_file)
    dfs = coco_to_dataframes(coco, images_folder)

    images_df = dfs.get("images")
    annotations_df = dfs.get("annotations")

    # --- 2. Explorer le dataset ---
    explore_dataset(images_df, annotations_df, images_folder)

    # --- 2b. Détecter les annotations orphelines ---
    orphan_annotations = annotations_without_images(annotations_df, images_df)
    if not orphan_annotations.empty:
        print(f"[INFO] {len(orphan_annotations)} annotations orphelines détectées")
        print(orphan_annotations[["id", "image_id"]])

    # --- 3. Nettoyer le dataset ---
    print("[START] Nettoyage du dataset...")
    images_df_clean, annotations_df_clean = clean_dataset(
        images_df, annotations_df, images_folder
    )

    # --- 4. Mettre à jour les DataFrames dans dfs ---
    dfs["images"] = images_df_clean
    dfs["annotations"] = annotations_df_clean

    # --- 5. Sauvegarder le JSON nettoyé ---
    save_coco_annotations(coco, dfs, output_file)

    print("[END] Nettoyage terminé ✅")
