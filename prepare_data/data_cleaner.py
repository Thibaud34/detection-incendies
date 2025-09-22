import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path

from data_cleaner import (
    get_file_extensions,
    check_images_consistency,
    images_without_annotations,
    annotations_without_images,
    detect_abnormal_annotations,
    remove_images_without_annotations,
    fix_bbox,
    correct_bboxes,
    clean_dataset,
)

# -------------------------------------------------------
# Tests get_file_extensions
# -------------------------------------------------------
def test_get_file_extensions_ok(tmp_path):
    # Créer fichiers temporaires
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.png").touch()
    extensions = get_file_extensions(str(tmp_path))
    assert set(extensions) == {".jpg", ".png"}

def test_get_file_extensions_folder_not_exist():
    with pytest.raises(FileNotFoundError):
        get_file_extensions("not_a_real_folder")

# -------------------------------------------------------
# Tests check_images_consistency
# -------------------------------------------------------
def test_check_images_consistency(tmp_path):
    # Fichier déclaré mais pas présent
    df_images = pd.DataFrame([{"file_name": "img1.jpg"}])
    stats = check_images_consistency(df_images, tmp_path)
    assert stats["total_declared"] == 1
    assert stats["missing_count"] == 1
    assert stats["unreferenced_count"] == 0

# -------------------------------------------------------
# Tests images_without_annotations
# -------------------------------------------------------
def test_images_without_annotations_some_missing():
    df_images = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
    ])
    df_annotations = pd.DataFrame([
        {"id": 10, "image_id": 1}
    ])
    result = images_without_annotations(df_images, df_annotations)
    assert len(result) == 1
    assert result.iloc[0]["file_name"] == "img2.jpg"

def test_images_without_annotations_with_filepath(tmp_path):
    df_images = pd.DataFrame([{"id": 1, "file_name": "img1.jpg"}])
    df_annotations = pd.DataFrame(columns=["image_id"])
    result = images_without_annotations(df_images, df_annotations, images_dir=tmp_path)
    assert "file_path" in result.columns
    assert result.iloc[0]["file_path"].endswith("img1.jpg")

# -------------------------------------------------------
# Tests annotations_without_images
# -------------------------------------------------------
def test_annotations_without_images_orphans():
    df_images = pd.DataFrame([{"id": 1}])
    df_annotations = pd.DataFrame([
        {"id": 10, "image_id": 1},
        {"id": 20, "image_id": 2},
    ])
    result = annotations_without_images(df_annotations, df_images)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == 2

# -------------------------------------------------------
# Tests detect_abnormal_annotations
# -------------------------------------------------------
def test_detect_abnormal_annotations():
    df_annotations = pd.DataFrame([
        {"id": 1, "bbox": [0, 0, 50, 50]},   # normal
        {"id": 2, "bbox": [0, 0, 0, 30]},    # largeur = 0
        {"id": 3, "bbox": [0, 0, 30, -5]},   # hauteur négative
    ])
    result = detect_abnormal_annotations(df_annotations)
    assert set(result["id"]) == {2, 3}

# -------------------------------------------------------
# Tests fix_bbox + correct_bboxes
# -------------------------------------------------------
def test_fix_bbox_inside_bounds():
    row = {"bbox": [-5, -5, 20, 20]}
    fixed = fix_bbox(row, img_w=100, img_h=100)
    assert fixed[0] >= 0
    assert fixed[1] >= 0

def test_correct_bboxes_fixing():
    df_images = pd.DataFrame([{"id": 1, "width": 100, "height": 100}])
    df_annotations = pd.DataFrame([
        {"id": 1, "image_id": 1, "bbox": [-5, -5, 200, 200]}
    ])
    fixed = correct_bboxes(df_images, df_annotations)
    assert fixed.iloc[0]["bbox"][0] >= 0
    assert fixed.iloc[0]["bbox"][2] <= 100

# -------------------------------------------------------
# Tests remove_images_without_annotations + clean_dataset
# -------------------------------------------------------
def test_remove_images_without_annotations(tmp_path):
    # Préparer fichiers image
    img1 = tmp_path / "img1.jpg"
    img1.touch()
    img2 = tmp_path / "img2.jpg"
    img2.touch()

    df_images = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
    ])
    df_annotations = pd.DataFrame([{"image_id": 1}])

    result = remove_images_without_annotations(df_images, df_annotations, str(tmp_path))
    # img2 supprimée
    assert len(result) == 1
    assert not img2.exists()

def test_clean_dataset_pipeline(tmp_path):
    img = tmp_path / "img1.jpg"
    img.touch()
    df_images = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}
    ])
    df_annotations = pd.DataFrame([
        {"id": 10, "image_id": 1, "bbox": [-5, -5, 200, 200]}
    ])
    new_images, new_annotations = clean_dataset(df_images, df_annotations, str(tmp_path))
    assert len(new_images) == 1
    assert new_annotations.iloc[0]["bbox"][0] >= 0
