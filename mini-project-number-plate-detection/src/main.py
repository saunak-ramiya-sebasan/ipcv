from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from preprocessing import preprocess_image
from plate_detection import apply_canny, find_contours, get_plate_contour, crop_plate
from ocr import extract_text


def process_single_image(image_path: Path) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Process a single image through the full ANPR pipeline:
    1. Load raw image
    2. Preprocess (grayscale + blur)
    3. Detect plate edges
    4. Find and crop plate contour
    5. Extract text with OCR

    Returns: (extracted_text, detected_plate_image)
    """
    print(f"\nProcessing: {image_path.name}")
    print("-" * 40)

    original = cv2.imread(str(image_path))
    if original is None:
        print(f"  Error: Could not load image")
        return None, None
    print(f"  Loaded image: {original.shape[1]}x{original.shape[0]}")

    try:
        gray, blurred = preprocess_image(image_path)
        print(f"  Preprocessed: grayscale + gaussian blur")
    except Exception as e:
        print(f"  Error preprocessing: {e}")
        return None, None

    edges = apply_canny(blurred)
    print(f"  Edge detection: Canny applied")

    contours = find_contours(edges)
    print(f"  Found {len(contours)} contours")

    plate_contour = get_plate_contour(contours)

    if plate_contour is None:
        print(f"  Detection: No plate found")
        return None, None

    print(f"  Detection: Plate contour found")

    plate_img = crop_plate(original, plate_contour)
    print(f"  Cropped plate: {plate_img.shape[1]}x{plate_img.shape[0]}")

    text = extract_text(plate_img)
    print(f"  OCR result: '{text}'")

    return text, plate_img


def run_pipeline(raw_dir: Path = Path("dataset/raw"), limit: int = 10) -> None:
    """
    Run the ANPR pipeline on images in the raw directory.
    """
    if not raw_dir.exists():
        print(f"Raw directory not found: {raw_dir}")
        print("Run: uv run src/dataset_loader.py first")
        return

    image_files = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {raw_dir}")
        return

    test_images = image_files[:limit]
    print(f"=" * 50)
    print(f"ANPR Pipeline Test")
    print(f"Testing on {len(test_images)} images")
    print(f"=" * 50)

    success_count = 0
    results: list[Tuple[str, str]] = []

    for img_path in test_images:
        text, plate_img = process_single_image(img_path)

        if text:
            results.append((img_path.name, text))
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"Summary: {success_count}/{len(test_images)} images processed successfully")
    print(f"{'=' * 50}")

    if results:
        print("\nResults:")
        for name, text in results:
            print(f"  {name}: {text}")


if __name__ == "__main__":
    run_pipeline(limit=10)
