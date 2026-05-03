import os
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "ud-smart-city/license-plate-dataset"
RAW_DIR = Path("dataset/raw")


def load_and_save_dataset(limit: int = 10) -> None:
    """Load dataset from Hugging Face and save images to disk."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split="train", token=hf_token)

    saved = 0
    for i, row in enumerate(dataset):
        if i >= limit:
            break

        image = row["image"]
        image_id = row.get("id", f"img_{i:04d}")

        if hasattr(image, "filename"):
            filename = Path(image.filename).name
        else:
            filename = f"{image_id}.jpg"

        output_path = RAW_DIR / filename

        image_array = np.array(image)

        filename = f"{image_id}.jpg"
        output_path = RAW_DIR / filename

        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_array)

        saved += 1

    print(f"Saved {saved} images to {RAW_DIR}")


if __name__ == "__main__":
    load_and_save_dataset(limit=120)
