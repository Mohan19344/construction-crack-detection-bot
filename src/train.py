"""Train a simple crack/no-crack image classifier."""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(image_size: int = 180) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Dataset root folder")
    parser.add_argument("--output", default="models/crack_classifier.keras")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    for folder in (train_dir / "crack", train_dir / "no_crack",
                   val_dir / "crack", val_dir / "no_crack"):
        if not folder.exists() or not any(folder.iterdir()):
            raise SystemExit(
                f"Missing images in {folder}. Add labeled images first; "
                "see the dataset section in README.md."
            )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, label_mode="binary", seed=42
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir, image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, label_mode="binary", shuffle=False
    )

    # image_dataset_from_directory assigns labels alphabetically:
    # crack=0, no_crack=1. We want sigmoid output to mean crack probability,
    # so explicitly invert labels during training and validation.
    def crack_positive(images, labels):
        return images, 1.0 - labels

    train_ds = train_ds.map(crack_positive).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(crack_positive).prefetch(tf.data.AUTOTUNE)

    model = build_model(args.image_size)
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nSaved model to: {output_path}")
    print("Next: test on images that were not used for training or validation.")


if __name__ == "__main__":
    main()
