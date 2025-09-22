# tests/test_data_cleaner.py

import sys
from pathlib import Path
import pandas as pd
import pytest

# --- le dossier parent pour que Python trouve data_cleaner.py ---
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from prepare_data.data_cleaner import (
    get_file_extensions,
    check_images_consistency,
    images_without_annotations,
    annotations_without_images,
    detect_abnormal_annotations
)

# ------------------------------
# 1/ Tests pour get_file_extensions:
# * Test avec un dossier vide → doit renvoyer []
# * Test avec des fichiers .jpg, .png, .JPG => doit renvoyer {".jpg", ".png"} (extensions en minuscule et uniques)
# ------------------------------
def test_get_file_extensions_empty(tmp_path: Path):
    # Dossier vide
    result = get_file_extensions(str(tmp_path))
    assert result == []

def test_get_file_extensions_with_files(tmp_path: Path):
    # Créer des fichiers factices
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.JPG").touch()
    
    result = get_file_extensions(str(tmp_path))
    # Les extensions doivent être uniques et en minuscule
    assert set(result) == {".jpg", ".png"}

# ------------------------------
# 2/ Tests pour check_images_consistency: 
# * Vérifie que le nombre de fichiers déclarés vs réels est correct
# * Vérifie le nombre de fichiers manquants et non référencés
# ------------------------------
def test_check_images_consistency(tmp_path: Path):
    # Fichiers réels
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()
    
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
        {"id": 3, "file_name": "img3.jpg"}  # manquant
    ])
    
    stats = check_images_consistency(images_df, str(tmp_path))
    assert stats["total_declared"] == 3
    assert stats["total_actual"] == 2
    assert stats["missing_count"] == 1
    assert stats["unreferenced_count"] == 0

# ------------------------------
# 3/ Tests pour images_without_annotations: 
# * Vérifie que les images sans annotations sont correctement identifiées
# ------------------------------
def test_images_without_annotations_basic():
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
        {"id": 3, "file_name": "img3.jpg"},
    ])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1},
    ])
    
    result = images_without_annotations(images_df, annotations_df)
    assert len(result) == 2
    assert set(result["file_name"]) == {"img2.jpg", "img3.jpg"}

# ------------------------------
#4/Tests pour annotations_without_images: 
# * Vérifie que seules les annotations orphelines sont retournées
# ------------------------------
def test_annotations_without_images_basic():
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
    ])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1},
        {"id": 11, "image_id": 2},  # orpheline
    ])
    
    result = annotations_without_images(annotations_df, images_df)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == 2

# ------------------------------
# 5/ Tests pour detect_abnormal_annotations:
# * Vérifie que seules les annotations avec des bboxes aberrantes sont détectées
# ------------------------------
def test_detect_abnormal_annotations_basic():
    annotations_df = pd.DataFrame([
        {"id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "bbox": [0, 0, 0, 10]},  # anormale
        {"id": 3, "bbox": [5, 5, -5, 5]},  # anormale
    ])
    
    result = detect_abnormal_annotations(annotations_df)
    assert len(result) == 2
    assert set(result["id"]) == {2, 3}

# ------------------------------
#  pytest : cmd terminal
# ------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])
