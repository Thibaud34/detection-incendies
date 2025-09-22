from pathlib import Path
from typing import Set, List
import pandas as pd

# fonction pour obtenir les extensions de fichiers dans un dossier:

def get_file_extensions(folder: str) -> Set[str]:
    """
    Retourne l'ensemble des extensions de fichiers présentes dans le dossier `folder`.

    Args:
        folder (str): Chemin vers le dossier à explorer.

    Returns:
        Set[str]: Ensemble des extensions trouvées, en minuscules.
                Exemples : {'.jpg', '.png', '.jpeg'}
    """
    p = Path(folder)# Crée un objet Path pour le dossier.
    
    # Vérifier que le dossier existe
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Le dossier spécifié n'existe pas : {folder}")
    
    # Compréhension de liste pour récupérer les extensions des fichiers
    extensions = {f.suffix.lower() for f in p.iterdir() if f.is_file()} 
    
    return extensions
#p.iterdir() : liste tous les éléments (fichiers + dossiers) du dossier.

#if f.is_file() : on ne garde que les fichiers.

#f.suffix.lower() : récupère l’extension du fichier en minuscules.

#L’ensemble (set) supprime automatiquement les doublons.

# -------------------------
# Vérifications images / annotations
# -------------------------

def check_images_on_disk(images_df: pd.DataFrame, images_folder: str) -> List[str]:
    """
    Vérifie la cohérence entre les images listées dans images_df et les fichiers présents physiquement.
    
    Args:
        images_df (pd.DataFrame): DataFrame contenant au moins la colonne 'file_name'.
        images_folder (str): Chemin vers le dossier contenant les images.
    
    Returns:
        List[str]: Liste des images manquantes sur le disque.
    """
    import os
    missing_images = [f for f in images_df['file_name'] if not os.path.exists(os.path.join(images_folder, f))]
    return missing_images


def get_images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renvoie un DataFrame des images qui n'ont aucune annotation.
    
    Args:
        images_df (pd.DataFrame): DataFrame des images, avec au moins la colonne 'id'.
        annotations_df (pd.DataFrame): DataFrame des annotations, avec au moins la colonne 'image_id'.
    
    Returns:
        pd.DataFrame: Sous-DataFrame des images sans annotations.
    """
    # Obtenir la liste des ids uniques présents dans les annotations
    annotated_ids = annotations_df['image_id'].unique()
    
    # Filtrer le DataFrame des images pour ne garder que celles qui ne sont pas annotées
    images_no_annotations = images_df[~images_df['id'].isin(annotated_ids)].reset_index(drop=True)
    
    return images_no_annotations


