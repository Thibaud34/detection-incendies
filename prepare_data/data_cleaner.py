from pathlib import Path
import os
from typing import Optional, Union
import pandas as pd


# =================== Gestion des fichiers ===================
def get_file_extensions(folder_path: str):
    """
    Retourne la liste des extensions uniques des fichiers dans le dossier donné.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Le dossier {folder_path} est introuvable.")

    extensions = list({f.suffix.lower() for f in folder.iterdir() if f.is_file() and f.suffix})
    return extensions


def check_images_consistency(images_df, images_dir):
    """
    Vérifie la cohérence entre les images du JSON COCO et celles présentes physiquement.
    """
    declared_files = set(images_df["file_name"].tolist())
    actual_files = set(os.listdir(images_dir))
    missing_files = declared_files - actual_files
    unreferenced_files = actual_files - declared_files

    return {
        "total_declared": len(declared_files),
        "total_actual": len(actual_files),
        "missing_count": len(missing_files),
        "unreferenced_count": len(unreferenced_files)
    }


# =================== Nettoyage des données ===================
def images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame,
                               images_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Retourne les images sans aucune annotation.
    """
    if "id" not in images_df.columns or "file_name" not in images_df.columns:
        raise ValueError("images_df doit contenir 'id' et 'file_name'")
    if "image_id" not in annotations_df.columns:
        raise ValueError("annotations_df doit contenir 'image_id'")

    annotated_ids = set(annotations_df["image_id"].astype(str).unique())
    mask_no_ann = ~images_df["id"].astype(str).isin(annotated_ids)
    result = images_df[mask_no_ann].copy()

    if images_dir is not None:
        folder = Path(images_dir)
        result["file_path"] = result["file_name"].apply(lambda fn: str(folder / fn))

    return result


def annotations_without_images(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les annotations dont l'image n'existe pas.
    """
    valid_image_ids = set(images_df["id"].astype(str).unique())
    orphan_ann = annotations_df[~annotations_df["image_id"].astype(str).isin(valid_image_ids)]
    return orphan_ann


def detect_abnormal_annotations(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les bounding boxes invalides.
    """
    df = annotations_df.copy()
    df["bbox_width"] = df["bbox"].apply(lambda b: b[2])
    df["bbox_height"] = df["bbox"].apply(lambda b: b[3])

    abnormal = df[
        (df["bbox_width"] <= 0) |
        (df["bbox_height"] <= 0) |
        ((df["bbox_width"] == 0) & (df["bbox_height"] != 0)) |
        ((df["bbox_height"] == 0) & (df["bbox_width"] != 0))
    ]
    return abnormal


# =================== Correction des bounding boxes ===================
def fix_bbox(row, img_w, img_h):
    """
    Corrige une bounding box pour qu'elle soit contenue dans l'image.
    """
    x, y, w, h = row['bbox']
    x_max = min(x + w, img_w)
    y_max = min(y + h, img_h)
    x = max(x, 0)
    y = max(y, 0)
    w = max(1, x_max - x)
    h = max(1, y_max - y)
    return [x, y, w, h]

# =================== Correction ciblée des bounding boxes ===================
def correct_bboxes(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Corrige uniquement les bounding boxes invalides (négatives ou hors dimensions).
    """
    annotations_df = annotations_df.copy()

    # 1️⃣ Détection des invalides (mêmes règles que dans explore_dataset)
    ann_with_img = annotations_df.merge(
        images_df[["id", "width", "height"]],
        left_on="image_id",
        right_on="id",
        suffixes=("_ann", "_img")
    )
    ann_with_img["x_min"] = ann_with_img["bbox"].apply(lambda b: b[0])
    ann_with_img["y_min"] = ann_with_img["bbox"].apply(lambda b: b[1])
    ann_with_img["x_max"] = ann_with_img.apply(lambda row: row["bbox"][0] + row["bbox"][2], axis=1)
    ann_with_img["y_max"] = ann_with_img.apply(lambda row: row["bbox"][1] + row["bbox"][3], axis=1)

    invalid_ann = ann_with_img[
        (ann_with_img["x_min"] < 0) |
        (ann_with_img["y_min"] < 0) |
        (ann_with_img["x_max"] > ann_with_img["width"]) |
        (ann_with_img["y_max"] > ann_with_img["height"])
    ]

    # 2️⃣ Correction uniquement de celles marquées invalides
    corrected = 0
    for _, ann in invalid_ann.iterrows():
        img_w, img_h = ann["width"], ann["height"]
        old_bbox = ann["bbox"]

        new_bbox = fix_bbox({"bbox": old_bbox}, img_w, img_h)

        if any(a != b for a, b in zip(old_bbox, new_bbox)):
            corrected += 1
            # ✅ Forcer l'affectation comme objet unique
            idx = annotations_df.index[annotations_df["id"] == ann["id_ann"]][0]
            annotations_df.at[idx, "bbox"] = new_bbox

    return annotations_df, corrected




# =================== Pipeline de nettoyage ===================
def clean_dataset(images_df: pd.DataFrame, annotations_df: pd.DataFrame, images_dir: str):
    """
    Nettoie le dataset : supprime images sans annotations, annotations orphelines,
    corrige les bounding boxes et supprime les anomalies restantes.
    Retourne images_df_clean, annotations_df_clean et log détaillé.
    """
    log = {}

    # 1️⃣ Images sans annotations
    images_no_ann = images_without_annotations(images_df, annotations_df, images_dir)
    log["images_removed_no_annotations"] = len(images_no_ann)
    images_df_clean = images_df[~images_df["id"].isin(images_no_ann["id"])].copy()

    # 2️⃣ Annotations orphelines
    annotations_orphan = annotations_without_images(annotations_df, images_df_clean)
    log["annotations_orphan_removed"] = len(annotations_orphan)
    annotations_df_clean = annotations_df[~annotations_df["id"].isin(annotations_orphan["id"])].copy()

    # 3️⃣ Corriger les bounding boxes
    annotations_df_clean, corrected_count = correct_bboxes(images_df_clean, annotations_df_clean)
    log["annotations_bbox_corrected"] = corrected_count

    # 4️⃣ Supprimer anomalies restantes
    abnormal = detect_abnormal_annotations(annotations_df_clean)
    log["annotations_abnormal_removed"] = len(abnormal)
    annotations_df_clean = annotations_df_clean[~annotations_df_clean["id"].isin(abnormal["id"])].copy()

    return images_df_clean, annotations_df_clean, log
