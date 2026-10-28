# Construction Crack Detection Bot

A starter computer-vision project for detecting visible cracks in concrete images. This repository begins with a PC-based AI model workflow and can later be integrated with an ESP32-CAM robot.

> **Current status:** Starter code only. The model must be trained with a suitable dataset and tested before claiming reliable crack detection. This is not a structural-safety inspection tool.

## Project plan

1. Collect and label concrete images as `crack` or `no_crack`.
2. Train a small TensorFlow/Keras image classifier on a computer.
3. Run predictions on test images.
4. Later, capture images using the ESP32-CAM and send them to the computer for inference.
5. Explore edge deployment only after choosing hardware and measuring model size/performance.

## Requirements

- Python 3.10 or 3.11 recommended
- A computer with internet access for installing packages and obtaining a dataset
- Labeled images for both classes
- Optional: ESP32-CAM for image capture

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Ubuntu/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Prepare your dataset

Create this structure and put real images in the matching folders:

```text
dataset/
  train/
    crack/
    no_crack/
  validation/
    crack/
    no_crack/
```

Use a dataset you are allowed to use. Keep training and validation images separate; avoid placing near-duplicate images in both folders. Aim for a balanced set of examples from varied lighting, distances, concrete textures, and crack widths.

## Train

```bash
python src/train.py --data dataset --output models/crack_classifier.keras
```

The script saves a trained model and reports validation accuracy. Accuracy alone does not prove that the model is reliable; inspect false positives and false negatives on unseen images.

## Predict on an image

```bash
python src/predict.py --model models/crack_classifier.keras --image path/to/test_image.jpg
```

## ESP32-CAM integration (next phase)

The ESP32-CAM can capture or stream images, but this starter model runs on a computer. The initial integration plan is:
1. ESP32-CAM captures an image.
2. The image is sent to a laptop/Raspberry Pi over Wi-Fi.
3. The Python model predicts `crack` or `no_crack`.
4. The result is logged or displayed.

Do not assume the TensorFlow model can run directly on an ESP32-CAM. Direct deployment requires a compatible, quantized TinyML model and careful memory/performance testing.

## Repository structure

```text
construction-crack-detection-bot/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   └── README.md
├── models/
│   └── README.md
├── images/
│   └── README.md
└── src/
    ├── train.py
    └── predict.py
```

## Limitations

- This starter does not include a trained model or dataset.
- A classifier labels the whole image; it does not draw precise crack boundaries.
- Results depend on dataset quality and image conditions.
- This project is for learning and prototype development, not professional structural assessment.
