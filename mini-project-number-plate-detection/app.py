import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import streamlit as st

from src.ocr import extract_text
from src.plate_detection import (
    apply_canny,
    crop_plate,
    find_contours,
    get_plate_contour,
)

# from PIL import Image
from src.preprocessing import preprocess_image

st.set_page_config(
    page_title="ANPR System",
    page_icon="",
    layout="centered",
)

st.title("Automatic Number Plate Recognition")
st.markdown("Upload an image to detect and read license plates")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original is None:
        st.error("Failed to load image")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, use_container_width=True)

        with st.spinner("Processing..."):
            temp_path = Path("temp_upload.jpg")
            cv2.imwrite(str(temp_path), original)

            try:
                gray, blurred = preprocess_image(temp_path)
                edges = apply_canny(blurred)
                contours = find_contours(edges)
                plate_contour = get_plate_contour(contours)

                if plate_contour is not None:
                    plate_img = crop_plate(original, plate_contour)
                    text = extract_text(plate_img)

                    with col2:
                        st.subheader("Detected Plate")
                        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                        st.image(plate_rgb, use_container_width=True)

                    st.success(f"License Plate: {text}")
                else:
                    with col2:
                        st.subheader("Detected Plate")
                        st.info("No plate detected")

                    st.warning("Could not detect a license plate in this image")

            except Exception as e:
                st.error(f"Processing error: {e}")

            temp_path.unlink(missing_ok=True)

st.markdown("---")
st.caption("ANPR Pipeline | OpenCV + Tesseract OCR")
