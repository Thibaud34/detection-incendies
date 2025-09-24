# main.py

from prepare_data.pipeline import run_pipeline

if __name__ == "__main__":
    ## Fichiers et dossiers à adapter selon ton projet
    annotations_file = "data/_annotations.coco.json"   # chemin vers ton COCO JSON brut
    images_folder = "data/images"                # dossier contenant les images
    output_file = "data/annotations_clean.json"  # fichier de sortie nettoyé

    run_pipeline(annotations_file, images_folder, output_file)
