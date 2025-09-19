import pandas as pd


def count_images(images_df: pd.DataFrame) -> int:
    """
    Retourne le nombre total d'images.
    """
    return len(images_df)


def list_categories(categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne la liste des catégories disponibles.
    """
    return categories_df[["id", "name"]]


def annotations_statistics(annotations_df: pd.DataFrame, images_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Retourne les stats sur le nombre d'annotations par image.
    Si images_df est fourni, ajoute le nom de fichier.
    """
    stats = annotations_df.groupby("image_id").size().reset_index(name="nb_annotations")
    
    if images_df is not None:
        stats = stats.merge(images_df, left_on="image_id", right_on="id")
    
    return stats


def count_images_with_few_annotations(annotations_df: pd.DataFrame, images_df: pd.DataFrame, threshold: int = 3) -> int:
    """
    Retourne le nombre d'images qui ont moins de `threshold` annotations.
    """
    stats = annotations_df.groupby("image_id").size()
    images_less_than_threshold = stats[stats < threshold]
    return len(images_less_than_threshold)


def check_invalid_bounding_boxes(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie si les bounding boxes dépassent des dimensions d'image.
    Retourne un DataFrame avec uniquement les bounding boxes invalides.
    """
    # Joindre avec les dimensions des images
    ann_with_img = annotations_df.merge(
        images_df[["id", "width", "height"]],
        left_on="image_id",
        right_on="id",
        suffixes=("_ann", "_img")
    )

    # Calculer les coordonnées bbox
    ann_with_img["x_min"] = ann_with_img["bbox"].apply(lambda b: b[0])
    ann_with_img["y_min"] = ann_with_img["bbox"].apply(lambda b: b[1])
    ann_with_img["x_max"] = ann_with_img.apply(lambda row: row["bbox"][0] + row["bbox"][2], axis=1)
    ann_with_img["y_max"] = ann_with_img.apply(lambda row: row["bbox"][1] + row["bbox"][3], axis=1)

    # Filtrer les bbox invalides
    invalid_bboxes = ann_with_img[
        (ann_with_img["x_min"] < 0) |
        (ann_with_img["y_min"] < 0) |
        (ann_with_img["x_max"] > ann_with_img["width"]) |
        (ann_with_img["y_max"] > ann_with_img["height"])
    ]

    return invalid_bboxes
