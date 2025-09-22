import json
from pathlib import Path
import pandas as pd

def load_coco_annotations(file_path: str) -> dict:
    """
    Charge un fichier d'annotations COCO au format JSON.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def coco_to_dataframes(coco_data: dict, images_dir: str = None) -> dict[str, pd.DataFrame]:
    """
    Transforme un dictionnaire COCO en DataFrames Pandas et ajoute un chemin complet vers les images si fourni.

    Args:
        coco_data (dict): dictionnaire issu de load_coco_annotations.
        images_dir (str, optional): dossier contenant les images. Default=None

    Returns:
        dict[str, pd.DataFrame]: dictionnaire avec DataFrames pour images, annotations et catégories.
    """
    dfs = {}

    if "images" in coco_data:
        df_images = pd.DataFrame(coco_data["images"])
        if images_dir:
            df_images["file_path"] = df_images["file_name"].apply(lambda x: str(Path(images_dir) / x))
        dfs["images"] = df_images

    if "annotations" in coco_data:
        dfs["annotations"] = pd.DataFrame(coco_data["annotations"])

    if "categories" in coco_data:
        dfs["categories"] = pd.DataFrame(coco_data["categories"])
    
    return dfs

##########
def save_coco_annotations(coco_data: dict, dfs: dict[str, pd.DataFrame], output_path: str):
    """
    Sauvegarde un dictionnaire COCO mis à jour à partir de DataFrames dans un fichier JSON.

    Args:
        coco_data (dict): dictionnaire COCO original.
        dfs (dict[str, pd.DataFrame]): dictionnaire avec au moins "images" et "annotations".
        output_path (str): chemin du fichier de sortie.
    """
    if "images" in dfs:
        coco_data["images"] = dfs["images"].to_dict(orient="records")
    if "annotations" in dfs:
        coco_data["annotations"] = dfs["annotations"].to_dict(orient="records")
    if "categories" in dfs:
        coco_data["categories"] = dfs["categories"].to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False, default=int)

    print(f"[INFO] Fichier COCO sauvegardé → {output_path}")
