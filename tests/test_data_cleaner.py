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
# * Test avec un dossier vide => doit renvoyer []
# * Test avec des fichiers .jpg, .png, .JPG => doit renvoyer {".jpg", ".png"} (extensions en minuscule et uniques)
# * Test avec un fichier sans extension => doit renvoyer []
# * Test avec un chemin inexistant => doit lever FileNotFoundError
# ------------------------------
def test_get_file_extensions_empty(tmp_path: Path):
    """Cas : dossier vide → doit renvoyer une liste vide"""
    result = get_file_extensions(str(tmp_path))
    assert result == []

def test_get_file_extensions_with_files(tmp_path: Path):
    """Cas : différents fichiers → extensions uniques et en minuscule"""
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.JPG").touch()
    result = get_file_extensions(str(tmp_path))
    assert set(result) == {".jpg", ".png"}

def test_get_file_extensions_with_file_no_extension(tmp_path: Path):
    """Cas : fichier sans extension → ne doit pas planter"""
    (tmp_path / "readme").touch()
    result = get_file_extensions(str(tmp_path))
    assert result == []  # pas d'extension si fichier sans extension

def test_get_file_extensions_invalid_path():
    """Cas : chemin inexistant → doit lever une FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        get_file_extensions("chemin/inexistant")

# ------------------------------
# 2/ Tests pour check_images_consistency: 
# * Vérifie que le nombre de fichiers déclarés vs réels est correct
# * Vérifie le nombre de fichiers manquants
# * Vérifie le nombre de fichiers non référencés
# ------------------------------
def test_check_images_consistency(tmp_path: Path):
    """Cas : 1 image manquante"""
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
        {"id": 3, "file_name": "img3.jpg"}  # manquan
    ])
    stats = check_images_consistency(images_df, str(tmp_path))
    assert stats["total_declared"] == 3
    assert stats["total_actual"] == 2
    assert stats["missing_count"] == 1
    assert stats["unreferenced_count"] == 0

def test_check_images_consistency_extra_file(tmp_path: Path):
    """Cas : fichier présent mais pas déclaré"""
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()  # non référencé
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"}
    ])
    stats = check_images_consistency(images_df, str(tmp_path))
    assert stats["total_declared"] == 1
    assert stats["total_actual"] == 2
    assert stats["missing_count"] == 0
    assert stats["unreferenced_count"] == 1


# ------------------------------
# 3/ Tests pour images_without_annotations: 
# * Vérifie que les images sans annotations sont correctement identifiées
# * Cas avec DataFrames vides
# ------------------------------
def test_images_without_annotations_basic():
    """Cas : certaines images annotées, d’autres non"""
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

def test_images_without_annotations_empty():
    """Cas : aucune image → doit renvoyer un DF vide"""
    images_df = pd.DataFrame([], columns=["id", "file_name"])
    annotations_df = pd.DataFrame([], columns=["id", "image_id"])
    result = images_without_annotations(images_df, annotations_df)
    assert result.empty

# ------------------------------
#4/Tests pour annotations_without_images: 
# * Vérifie que seules les annotations orphelines sont retournées
# * Cas où toutes les annotations sont orphelines
# ------------------------------
def test_annotations_without_images_basic():
    """Cas : annotation orpheline détectée"""
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
    ])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1},
        {"id": 11, "image_id": 2},  # c'est ou l'image_id 2 n'existe pas(orpheline)
    ])
    result = annotations_without_images(annotations_df, images_df)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == 2

def test_annotations_without_images_all_orphan():
    """Cas : toutes les annotations sont orphelines"""
    images_df = pd.DataFrame([], columns=["id", "file_name"])
    annotations_df = pd.DataFrame([
        {"id": 11, "image_id": 2},
        {"id": 12, "image_id": 3}
    ])
    result = annotations_without_images(annotations_df, images_df)
    assert len(result) == 2

# ------------------------------
# 5/ Tests pour detect_abnormal_annotations:
# * Vérifie que seules les annotations avec des bboxes aberrantes sont détectées
# * Cas avec uniquement des bboxes valides
# ------------------------------
def test_detect_abnormal_annotations_basic():
    """Cas : bboxes anormales détectées"""
    annotations_df = pd.DataFrame([
        {"id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "bbox": [0, 0, 0, 10]},   #c'est ou la largeur = 0
        {"id": 3, "bbox": [5, 5, -5, 5]},  # c'est ou la hauteur négative
    ])
    result = detect_abnormal_annotations(annotations_df)
    assert len(result) == 2
    assert set(result["id"]) == {2, 3}

def test_detect_abnormal_annotations_all_valid():
    """Cas : toutes les bboxes valides → doit renvoyer un DF vide"""
    annotations_df = pd.DataFrame([
        {"id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "bbox": [5, 5, 15, 15]},
    ])
    result = detect_abnormal_annotations(annotations_df)
    assert result.empty

# ------------------------------
#  pytest : cmd terminal
# ------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])
