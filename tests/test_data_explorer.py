# tests/test_data_explorer.py

import sys
from pathlib import Path
import pandas as pd
import pytest

# --- le dossier parent pour que Python trouve data_explorer.py ---
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from prepare_data.data_explorer import (
    count_images,
    list_categories,
    annotations_statistics,
    count_images_with_few_annotations,
    check_invalid_bounding_boxes,
    explore_dataset
)

# ------------------------------
# 1/ Tests pour count_images
# ------------------------------
def test_count_images_basic():
    images_df = pd.DataFrame([
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"}
    ])
    assert count_images(images_df) == 2

# ------------------------------
# 2/ Tests pour list_categories
# ------------------------------
def test_list_categories_basic():
    categories_df = pd.DataFrame([
        {"id": 1, "name": "fire"},
        {"id": 2, "name": "smoke"}
    ])
    result = list_categories(categories_df)
    assert list(result.columns) == ["id", "name"]
    assert len(result) == 2

# ------------------------------
# 3/ Tests pour annotations_statistics
# ------------------------------
def test_annotations_statistics_basic():
    images_df = pd.DataFrame([{"id": 1}, {"id": 2}])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1},
        {"id": 11, "image_id": 1},
        {"id": 12, "image_id": 2}
    ])
    stats = annotations_statistics(annotations_df, images_df)
    assert "nb_annotations" in stats.columns
    assert stats.loc[stats["image_id"] == 1, "nb_annotations"].values[0] == 2
    assert stats.loc[stats["image_id"] == 2, "nb_annotations"].values[0] == 1

# ------------------------------
# 4/ Tests pour count_images_with_few_annotations
# ------------------------------
def test_count_images_with_few_annotations_basic():
    images_df = pd.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1},
        {"id": 11, "image_id": 2},
        {"id": 12, "image_id": 2}
    ])
    # seuil = 2 => image 1 et 3 sont < 2 annotations
    result = count_images_with_few_annotations(annotations_df, images_df, threshold=2)
    assert result == 2

# ------------------------------
# 5/ Tests pour check_invalid_bounding_boxes
# ------------------------------
def test_check_invalid_bounding_boxes_basic():
    images_df = pd.DataFrame([
        {"id": 1, "width": 100, "height": 100},
        {"id": 2, "width": 50, "height": 50}
    ])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1, "bbox": [10, 10, 20, 20]},  # valide
        {"id": 11, "image_id": 1, "bbox": [-5, 0, 10, 10]},   # invalide x_min < 0
        {"id": 12, "image_id": 2, "bbox": [0, 0, 60, 10]}     # invalide x_max > width
    ])
    invalid = check_invalid_bounding_boxes(annotations_df, images_df)
    assert len(invalid) == 2
    assert set(invalid["id_ann"]) == {11, 12}  


# ------------------------------
# 6/ Test explore_dataset (capture print)
# ------------------------------
def test_explore_dataset_basic(capsys):
    images_df = pd.DataFrame([
        {"id": 1, "width": 100, "height": 100},
        {"id": 2, "width": 50, "height": 50}
    ])
    annotations_df = pd.DataFrame([
        {"id": 10, "image_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 11, "image_id": 1, "bbox": [-5, 0, 10, 10]},
        {"id": 12, "image_id": 2, "bbox": [0, 0, 60, 10]}
    ])
    explore_dataset(images_df, annotations_df, "data/images")
    captured = capsys.readouterr()
    assert "[INFO] Exploration du dataset..." in captured.out
    assert "Bounding boxes invalides détectées : 2" in captured.out

# ------------------------------
#  pytest : cmd terminal
# ------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])
