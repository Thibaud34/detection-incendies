from prepare_data.data_loader import load_coco_annotations, coco_to_dataframes
from pathlib import Path

# --- 1️⃣ Définir les chemins ---
json_path = Path("data/_annotations.coco.json")
images_dir = Path("data/images")
output_dir = Path("data/csv")  # dossier pour stocker les CSV
output_dir.mkdir(exist_ok=True)  # créer le dossier s'il n'existe pas

# --- 2️⃣ Charger le JSON COCO ---
coco_data = load_coco_annotations(json_path)

# --- 3️⃣ Transformer en DataFrames ---
dfs = coco_to_dataframes(coco_data, images_dir)

images_df = dfs["images"]
annotations_df = dfs["annotations"]
categories_df = dfs["categories"]

# --- 4️⃣ Afficher un aperçu ---
print("=== Images ===")
print(images_df.head())

print("=== Annotations ===")
print(annotations_df.head())

print("=== Categories ===")
print(categories_df.head())

# --- 5️⃣ Sauvegarder en CSV ---
images_df.to_csv(output_dir / "images.csv", index=False)
annotations_df.to_csv(output_dir / "annotations.csv", index=False)
categories_df.to_csv(output_dir / "categories.csv", index=False)

print(f"\nCSV générés dans {output_dir.resolve()}")
