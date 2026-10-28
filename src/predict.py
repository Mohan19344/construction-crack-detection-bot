"""Predict whether an image appears to contain a crack."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/crack_classifier.keras")
    parser.add_argument("--image", required=True, help="Path to a concrete image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold from 0 to 1")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train it first.")
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    model = keras.models.load_model(model_path)
    image_size = model.input_shape[1:3]
    image = Image.open(image_path).convert("RGB").resize(
        (image_size[1], image_size[0])
    )
    array = np.asarray(image, dtype=np.float32)[None, ...]
    crack_probability = float(model.predict(array, verbose=0)[0][0])

    print(f"Estimated crack probability: {crack_probability:.1%}")
    if crack_probability >= args.threshold:
        print("Prediction: CRACK-LIKE PATTERN")
    else:
        print("Prediction: NO CRACK-LIKE PATTERN DETECTED")
    print("Note: This is a prototype prediction, not a structural safety assessment.")


if __name__ == "__main__":
    main()
