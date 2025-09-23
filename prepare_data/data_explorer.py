# prepare_data/data_explorer.py

import pandas as pd
from prepare_data.data_cleaner import annotations_without_images

# --- Fonctions utilitaires ---

def count_images(images_df: pd.DataFrame) -> int:
    """Retourne le nombre d'images dans le DataFrame"""
    return len(images_df)

def list_categories(categories_df: pd.DataFrame) -> pd.DataFrame:
    """Liste les catégories présentes"""
    return categories_df[["id", "name"]]

def annotations_statistics(annotations_df: pd.DataFrame, images_df: pd.DataFrame = None) -> pd.DataFrame:
    """Statistiques d'annotations par image"""
    stats = annotations_df.groupby("image_id").size().reset_index(name="nb_annotations")
    if images_df is not None:
        stats = stats.merge(images_df, left_on="image_id", right_on="id")
    return stats

def count_images_with_few_annotations(annotations_df: pd.DataFrame, images_df: pd.DataFrame, threshold: int = 3) -> int:
    """Compte le nombre d'images avec moins de 'threshold' annotations"""
    stats = annotations_df.groupby("image_id").size().reindex(images_df["id"], fill_value=0)
    return (stats < threshold).sum()

def check_invalid_bounding_boxes(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte toutes les bounding boxes invalides ou hors limites, identiques à la logique de correct_bboxes.
    """
    ann_with_img = annotations_df.merge(
        images_df[["id", "width", "height"]],
        left_on="image_id",
        right_on="id",
        suffixes=("_ann", "_img")
    )

    # Extraire coordonnées et dimensions
    ann_with_img["x_min"] = ann_with_img["bbox"].apply(lambda b: b[0])
    ann_with_img["y_min"] = ann_with_img["bbox"].apply(lambda b: b[1])
    ann_with_img["x_max"] = ann_with_img.apply(lambda row: row["bbox"][0] + row["bbox"][2], axis=1)
    ann_with_img["y_max"] = ann_with_img.apply(lambda row: row["bbox"][1] + row["bbox"][3], axis=1)
    ann_with_img["width_bbox"] = ann_with_img["bbox"].apply(lambda b: b[2])
    ann_with_img["height_bbox"] = ann_with_img["bbox"].apply(lambda b: b[3])

    # Détection identique à correct_bboxes
    invalid = ann_with_img[
        (ann_with_img["x_min"] < 0) |
        (ann_with_img["y_min"] < 0) |
        (ann_with_img["x_max"] > ann_with_img["width"]) |
        (ann_with_img["y_max"] > ann_with_img["height"]) |
        (ann_with_img["width_bbox"] <= 0) |
        (ann_with_img["height_bbox"] <= 0)
    ]
    return invalid

# --- Fonction principale d'exploration ---
def explore_dataset(images_df: pd.DataFrame, annotations_df: pd.DataFrame, images_folder: str):
    """
    Explore le dataset et affiche des statistiques utiles pour l'analyse.
    Compatible avec la pipeline.
    """
    print("[INFO] Exploration du dataset...")

    # Nombre d'images et d'annotations
    print(f"- Nombre d'images : {count_images(images_df)}")
    print(f"- Nombre d'annotations : {len(annotations_df)}")

    # Statistiques d'annotations
    stats = annotations_statistics(annotations_df, images_df)
    print(f"- Moyenne d'annotations par image : {stats['nb_annotations'].mean():.2f}")

    # Images avec peu d'annotations
    few = count_images_with_few_annotations(annotations_df, images_df)
    print(f"- Images avec <3 annotations : {few}")

    # Bounding boxes invalides détectées
    invalid = check_invalid_bounding_boxes(annotations_df, images_df)
    print(f"- Bounding boxes invalides détectées : {len(invalid)}")

    # Annotations orphelines
    orphan_ann = annotations_without_images(annotations_df, images_df)
    print(f"- Annotations orphelines détectées : {len(orphan_ann)}")
    if not orphan_ann.empty:
        print(orphan_ann[["id", "image_id"]])

    print("[INFO] Exploration terminée ✅")
