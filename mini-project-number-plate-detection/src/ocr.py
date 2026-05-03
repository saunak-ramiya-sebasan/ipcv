from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract


def preprocess_plate(plate_image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """
    Preprocess plate image for OCR:
    1. Resize to consistent height
    2. Convert to grayscale
    3. Apply threshold
    """
    h, w = plate_image.shape[:2]
    scale = target_height / float(h)
    new_width = int(w * scale)

    resized = cv2.resize(plate_image, (new_width, target_height))

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def extract_text(plate_image: np.ndarray) -> str:
    """
    Extract text from plate image using Tesseract OCR.
    Uses PSM 7 (single text line) and whitelist A-Z, 0-9.
    """
    preprocessed = preprocess_plate(plate_image)

    config = (
        '--psm 7 '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    text = pytesseract.image_to_string(preprocessed, config=config)

    cleaned = text.strip().replace(' ', '').upper()

    return cleaned


def process_detected_plates(detected_dir: Path = Path("dataset/detected")) -> None:
    """Process all detected plate images and extract text."""
    if not detected_dir.exists():
        print(f"Detected plates directory not found: {detected_dir}")
        return

    plate_images = list(detected_dir.glob("*_plate.jpg"))
    if not plate_images:
        print(f"No plate images found in {detected_dir}")
        return

    print(f"Processing {len(plate_images)} detected plates...")
    print("-" * 40)

    results: list[tuple[str, str]] = []

    for plate_path in sorted(plate_images):
        try:
            plate_img = cv2.imread(str(plate_path))
            if plate_img is None:
                print(f"Failed to load: {plate_path.name}")
                continue

            text = extract_text(plate_img)
            results.append((plate_path.name, text))
            print(f"{plate_path.name}: '{text}'")

        except Exception as e:
            print(f"Error processing {plate_path.name}: {e}")

    print("-" * 40)
    valid_results = [r for r in results if r[1]]
    print(f"Extracted text from {len(valid_results)}/{len(plate_images)} plates")


if __name__ == "__main__":
    process_detected_plates()
