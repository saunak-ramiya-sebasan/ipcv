from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROCESSED_DIR = Path("dataset/processed")
DETECTED_DIR = Path("dataset/detected")


def apply_canny(
    image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150
) -> np.ndarray:
    """Apply Canny edge detection."""
    edges = cv2.Canny(image, low_thresh, high_thresh)
    return edges


def find_contours(edges: np.ndarray) -> List[np.ndarray]:
    """Find contours from edge map."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def filter_plate_contour(contour: np.ndarray) -> bool:
    """
    Filter contour based on plate characteristics:
    - 4-sided polygon
    - Aspect ratio between 2:1 and 5:1
    - Minimum area
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        return False

    x, y, w, h = cv2.boundingRect(approx)
    if h == 0:
        return False

    aspect_ratio = w / float(h)
    if not (2.0 <= aspect_ratio <= 5.0):
        return False

    min_area = 500
    if cv2.contourArea(contour) < min_area:
        return False

    return True


def get_plate_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """Sort contours by area and return best matching plate contour."""
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in sorted_contours:
        if filter_plate_contour(contour):
            return contour

    return None


def crop_plate(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Crop the plate region from the image using perspective transform."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_plate(
    image_path: Path, original_image: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect and crop license plate from preprocessed image.
    Returns (cropped_plate, debug_image).
    """
    blurred = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if blurred is None:
        return None, None

    if original_image is None:
        original_image = cv2.imread(
            str(image_path).replace("_blurred", "").replace("processed", "raw")
        )

    edges = apply_canny(blurred)
    contours = find_contours(edges)
    plate_contour = get_plate_contour(contours)

    debug_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    if plate_contour is not None:
        cv2.drawContours(debug_image, [plate_contour], -1, (0, 255, 0), 3)

        if original_image is not None:
            plate_img = crop_plate(original_image, plate_contour)
            return plate_img, debug_image

    return None, debug_image


def detect_all_plates() -> None:
    """Detect plates in all preprocessed images and save cropped plates."""
    DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    blurred_images = list(PROCESSED_DIR.glob("*_blurred.jpg"))
    if not blurred_images:
        print(f"No preprocessed images found in {PROCESSED_DIR}")
        return

    detected_count = 0
    for blurred_path in blurred_images:
        try:
            raw_path = Path(
                str(blurred_path).replace("_blurred", "").replace("processed", "raw")
            )
            original_image = cv2.imread(str(raw_path))

            if original_image is None:
                continue

            plate_img, debug_img = detect_plate(blurred_path, original_image)

            if plate_img is not None:
                output_path = (
                    DETECTED_DIR
                    / f"{blurred_path.stem.replace('_blurred', '_plate')}.jpg"
                )
                cv2.imwrite(str(output_path), plate_img)
                detected_count += 1
            else:
                print(f"No plate detected in {blurred_path.name}")

        except Exception as e:
            print(f"Error detecting plate in {blurred_path}: {e}")

    print(f"Detected {detected_count}/{len(blurred_images)} plates")


if __name__ == "__main__":
    detect_all_plates()
