from pathlib import Path
import os
from typing import Optional, Union
import pandas as pd


# def get_file_extensions(folder_path: str):
#     """
#     Retourne la liste des extensions uniques des fichiers dans le dossier donné.

#     Args:
#         folder_path (str): chemin du dossier

#     Returns:
#         list: extensions trouvées (ex: ['.jpg', '.png'])
#     """
#     folder = Path(folder_path)
#     if not folder.exists() or not folder.is_dir():
#         raise FileNotFoundError(f"Le dossier {folder_path} est introuvable.")

#     # Compréhension de liste pour récupérer toutes les extensions
#     extensions = list({f.suffix.lower() for f in folder.iterdir() if f.is_file()})
#     return extensions
def get_file_extensions(folder_path: str):
    """
    Retourne la liste des extensions uniques des fichiers dans le dossier donné.

    Args:
        folder_path (str): chemin du dossier

    Returns:
        list: extensions trouvées (ex: ['.jpg', '.png'])
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Le dossier {folder_path} est introuvable.")

    # Compréhension de liste pour récupérer toutes les extensions
    extensions = list({f.suffix.lower() for f in folder.iterdir() if f.is_file() and f.suffix})
    return extensions

##############################################################

def check_images_consistency(images_df, images_dir):
    """
    Vérifie la cohérence entre les images référencées dans le fichier COCO
    et celles disponibles physiquement dans le dossier.

    Args:
        images_df (pd.DataFrame): DataFrame contenant le champ "images" du JSON
        images_dir (str | Path): chemin vers le dossier contenant les images

    Returns:
        dict: statistiques de cohérence
    """
    # Fichiers déclarés dans le JSON
    declared_files = set(images_df["file_name"].tolist())

    # Fichiers réellement présents
    actual_files = set(os.listdir(images_dir))

    # Différences
    missing_files = declared_files - actual_files
    unreferenced_files = actual_files - declared_files

    return {
        "total_declared": len(declared_files),
        "total_actual": len(actual_files),
        "missing_count": len(missing_files),
        "unreferenced_count": len(unreferenced_files)
    }
############################################

def images_without_annotations(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    images_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Renvoie un DataFrame des images qui n'ont aucune annotation.

    Args:
        images_df (pd.DataFrame): DataFrame issu de "images" du JSON COCO (doit contenir 'id' et 'file_name').
        annotations_df (pd.DataFrame): DataFrame des annotations (doit contenir 'image_id').
        images_dir (str | Path, optional): si fourni, ajoute une colonne 'file_path' construite
                                        comme images_dir / file_name.

    Returns:
        pd.DataFrame: DataFrame (copie) des lignes d'images sans annotations. Vide si aucune.
    """
    if "id" not in images_df.columns:
        raise ValueError("images_df doit contenir la colonne 'id'.")
    if "file_name" not in images_df.columns:
        raise ValueError("images_df doit contenir la colonne 'file_name'.")
    if "image_id" not in annotations_df.columns:
        raise ValueError("annotations_df doit contenir la colonne 'image_id'.")

    # On compare en string pour éviter des soucis int/str
    annotated_ids = set(annotations_df["image_id"].astype(str).unique())

    mask_no_ann = ~images_df["id"].astype(str).isin(annotated_ids)
    result = images_df[mask_no_ann].copy()

    if images_dir is not None:
        folder = Path(images_dir)
        result["file_path"] = result["file_name"].apply(lambda fn: str(folder / fn))

    return result

##################
def annotations_without_images(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les annotations qui ne correspondent à aucune image présente dans images_df.

    Args:
        annotations_df (pd.DataFrame): DataFrame des annotations (avec colonne "image_id")
        images_df (pd.DataFrame): DataFrame des images (avec colonne "id")

    Returns:
        pd.DataFrame: annotations orphelines (dont l'image_id est absent de images_df)
    """
    valid_image_ids = set(images_df["id"].astype(str).unique())
    orphan_ann = annotations_df[~annotations_df["image_id"].astype(str).isin(valid_image_ids)]
    return orphan_ann

##################################
def detect_abnormal_annotations(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les annotations avec des valeurs aberrantes dans les bounding boxes.
    
    Args:
        annotations_df (pd.DataFrame): DataFrame des annotations (avec colonne "bbox")
    
    Returns:
        pd.DataFrame: DataFrame contenant uniquement les annotations aberrantes
    """
    # Extraire width et height à partir de la bbox
    annotations_df = annotations_df.copy()
    annotations_df["bbox_width"] = annotations_df["bbox"].apply(lambda b: b[2])
    annotations_df["bbox_height"] = annotations_df["bbox"].apply(lambda b: b[3])
    
    # Détection des anomalies
    abnormal = annotations_df[
        (annotations_df["bbox_width"] <= 0) |
        (annotations_df["bbox_height"] <= 0) |
        ((annotations_df["bbox_width"] == 0) & (annotations_df["bbox_height"] != 0)) |
        ((annotations_df["bbox_height"] == 0) & (annotations_df["bbox_width"] != 0))
    ]
    
    return abnormal