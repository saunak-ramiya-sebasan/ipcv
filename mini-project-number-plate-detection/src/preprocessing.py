from pathlib import Path

import cv2

RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")


def preprocess_image(image_path: Path) -> tuple:
    """
    Preprocess an image: grayscale + Gaussian blur.
    Returns (grayscale_img, blurred_img).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray, blurred


def preprocess_all_images() -> None:
    """Preprocess all images in dataset/raw and save to dataset/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))

    if not image_files:
        print(f"No images found in {RAW_DIR}")
        return

    for img_path in image_files:
        try:
            gray, blurred = preprocess_image(img_path)

            gray_path = PROCESSED_DIR / f"{img_path.stem}_gray.jpg"
            blurred_path = PROCESSED_DIR / f"{img_path.stem}_blurred.jpg"

            cv2.imwrite(str(gray_path), gray)
            cv2.imwrite(str(blurred_path), blurred)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Preprocessed {len(image_files)} images to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_all_images()
