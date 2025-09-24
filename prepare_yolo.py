import json
import os
import shutil
import random

# Chemins (paths):
coco_json_path = "data/annotations_clean.json"  
images_dir = "data/images"
labels_dir = "data/labels"

# Créer le dossier labels s'il n'existe pas
os.makedirs(labels_dir, exist_ok=True)

# Charger le JSON COCO
with open(coco_json_path) as f:#ouvrir le fichier json
    coco = json.load(f)

# Créer mapping image_id => info image pour accès rapide
image_dict = {img['id']: img for img in coco['images']}

# # --- Convert COCO to YOLO ---
for ann in coco['annotations']:                              
    img_info = image_dict[ann['image_id']]                   
    img_w, img_h = img_info['width'], img_info['height']         
    # Nom du fichier .txt correspondant à l'image (c'est necessaire pour que yolo puisse lire les annotations)
    file_name_txt = os.path.splitext(img_info['file_name'])[0] + ".txt"
    file_path_txt = os.path.join(labels_dir, file_name_txt)
    
    # COCO bbox : [x_top_left, y_top_left, width, height]
    x, y, w, h = ann['bbox']
    
    # Conversion en YOLO : coordonnées normalisées
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    # Classe unique : 0 une seule classe "incendie"
    class_id = 0
    
    # Ajouter  la ligne dans le fichier .txt 
    with open(file_path_txt, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print(f"Conversion COCO => YOLO terminée. Fichiers labels créés dans : {labels_dir}")

# --- Dataset split (70/20/10) ---
splits = {"train": 0.7, "val": 0.2, "test": 0.1}

# Create folders
for split in splits:
    os.makedirs(f"dataset/{split}/images", exist_ok=True)
    os.makedirs(f"dataset/{split}/labels", exist_ok=True)

# Get all images
all_images = [img["file_name"] for img in coco["images"]]
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * splits["train"])
n_val = int(n_total * splits["val"])

train_imgs = all_images[:n_train]
val_imgs = all_images[n_train:n_train+n_val]
test_imgs = all_images[n_train+n_val:]

def move_files(img_list, split):
    for img_file in img_list:
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"

        # Copy images
        shutil.copy(os.path.join(images_dir, img_file), f"dataset/{split}/images/{img_file}")

        # Copy labels (si existe)
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, f"dataset/{split}/labels/{label_file}")

move_files(train_imgs, "train")
move_files(val_imgs, "val")
move_files(test_imgs, "test")

print("Split terminé :")
print(f"- Train : {len(train_imgs)} images")
print(f"- Val   : {len(val_imgs)} images")
print(f"- Test  : {len(test_imgs)} images")
