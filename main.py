# main.py
from prepare_data.data_loader import load_coco_annotations, coco_to_dataframes
from prepare_data.data_explorer import (
    count_images,
    list_categories,
    annotations_statistics,
    count_images_with_few_annotations,
    check_invalid_bounding_boxes
)
from prepare_data.data_cleaner import get_file_extensions
from pathlib import Path
from prepare_data.data_cleaner import check_images_consistency
from prepare_data.data_cleaner import images_without_annotations
from prepare_data.data_cleaner import annotations_without_images
from prepare_data.data_cleaner import detect_abnormal_annotations
from PIL import Image


# --- Définir les chemins ---
json_path = Path("data/_annotations.coco.json")
images_dir = Path("data/images")
output_dir = Path("data/csv")  # dossier pour stocker les CSV
output_dir.mkdir(exist_ok=True)  # créer le dossier s'il n'existe pas

# --- Charger le JSON COCO ---
coco_data = load_coco_annotations(json_path)

# --- Transformer en DataFrames ---
dfs = coco_to_dataframes(coco_data, images_dir)

images_df = dfs["images"]
annotations_df = dfs["annotations"]
categories_df = dfs["categories"]

# # --- Afficher un aperçu ---
# print("=== Images ===")
# print(images_df.head())

# print("=== Annotations ===")
# print(annotations_df.head())

# print("=== Categories ===")
# print(categories_df.head())

# # --- Sauvegarder en CSV ---
# images_df.to_csv(output_dir / "images.csv", index=False)
# annotations_df.to_csv(output_dir / "annotations.csv", index=False)
# categories_df.to_csv(output_dir / "categories.csv", index=False)

# print(f"\nCSV générés dans {output_dir.resolve()}")


# # --- Exploration / affichage ---
# print("=== Nombre total d'images ===")
# print(count_images(images_df))

# print("\n=== Catégories ===")
# print(list_categories(categories_df))

# print("\n=== Stats sur les annotations ===")
# stats = annotations_statistics(annotations_df, images_df)
# print(stats.head())

# print("\n=== Nombre d'images avec moins de 3 annotations ===")
# print(count_images_with_few_annotations(annotations_df, images_df))

# print("\n=== Bounding boxes invalides ===")
# invalid_bboxes = check_invalid_bounding_boxes(annotations_df, images_df)
# print(f"Nombre de bbox invalides : {len(invalid_bboxes)}")
# if len(invalid_bboxes) > 0:
#     print(invalid_bboxes.head())

# print("\n=== Extensions des fichiers dans le dossier images ===")
# extensions = get_file_extensions(images_dir)
# print(extensions)

# ################
# images_dir = "data/images"
# stats = check_images_consistency(images_df, images_dir)

# print("=== Vérification cohérence images ===")
# print(f"Images déclarées dans JSON : {stats['total_declared']}")
# print(f"Images présentes sur disque : {stats['total_actual']}")
# print(f"Images manquantes : {stats['missing_count']}")
# print(f"Images non référencées : {stats['unreferenced_count']}")

#################
# Récupérer les images sans annotations
no_ann = images_without_annotations(images_df, annotations_df, images_dir="data/images")

print("Nombre d'images sans annotations :", len(no_ann))

if not no_ann.empty:
    first_img_path = no_ann.iloc[0]["file_path"]
    print("Exemple d'image sans annotation :", first_img_path)

    # Ouvrir l’image avec PIL
    img = Image.open(first_img_path)
    img.show()

#######################
# orphans = annotations_without_images(annotations_df, images_df)

# print("Nombre d'annotations sans image :", len(orphans))
# if len(orphans) > 0:
#     print(orphans.head())
# #####################

# abnormal_ann = detect_abnormal_annotations(annotations_df)

# print("Nombre d'annotations aberrantes :", len(abnormal_ann))
# if len(abnormal_ann) > 0:
#     print(abnormal_ann.head())