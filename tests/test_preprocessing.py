import cv2
import numpy as np

from src.preprocessing import crop_to_mask, gray_world_lab_normalize


def test_crop_to_mask_matches_bbox_with_padding():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:70] = 255  # foreground rows 20-39, cols 30-69

    cropped, cropped_mask, status = crop_to_mask(image, mask, padding_frac=0.1)

    assert status == "cropped"
    # bbox height=20, width=40 -> padding = 2 rows, 4 cols per side
    expected_h = (40 - 20) + 2 * 2
    expected_w = (70 - 30) + 2 * 4
    assert cropped.shape[:2] == (expected_h, expected_w)
    assert cropped_mask.shape == cropped.shape[:2]


def test_crop_to_mask_clips_padding_to_image_bounds():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[0:10, 0:10] = 255  # touches top-left corner, padding should clip

    cropped, cropped_mask, status = crop_to_mask(image, mask, padding_frac=0.5)

    assert status == "cropped"
    assert cropped.shape[0] <= 50 and cropped.shape[1] <= 50
    assert cropped_mask.shape == cropped.shape[:2]


def test_crop_to_mask_empty_mask_falls_back_to_full_image():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)  # all background

    cropped, cropped_mask, status = crop_to_mask(image, mask)

    assert status == "mask_empty"
    assert cropped.shape == image.shape
    assert cropped_mask.shape == mask.shape


def test_gray_world_lab_normalize_reduces_colour_cast():
    base = np.full((40, 40, 3), 120, dtype=np.uint8)
    casted = base.copy()
    casted[..., 0] = np.clip(casted[..., 0].astype(int) + 60, 0, 255).astype(np.uint8)  # heavy red cast

    lab_before = cv2.cvtColor(casted, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_before, b_before = lab_before[..., 1].mean(), lab_before[..., 2].mean()

    corrected = gray_world_lab_normalize(casted)

    lab_after = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_after, b_after = lab_after[..., 1].mean(), lab_after[..., 2].mean()

    assert abs(a_after - 128) < abs(a_before - 128)
    assert abs(b_after - 128) < abs(b_before - 128)
