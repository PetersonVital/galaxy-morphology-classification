from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


MODEL_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
INFERENCE_RESULTS_PATH = Path("outputs/predictions/inference_results.csv")

SAMPLE_IMAGES_DIR = Path("data/sample_images")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")

GRADCAM_REPORT_PATH = REPORTS_DIR / "gradcam_report.csv"
POWERBI_GRADCAM_REPORT_PATH = POWERBI_DIR / "gradcam_report.csv"

IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_IMAGES_TO_PROCESS = 6


def ensure_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.as_posix()}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH.as_posix()}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model, class_names


def get_feature_extractor(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError("Could not find the feature extractor submodel.")


def build_classifier_head(model, feature_extractor):
    start_index = model.layers.index(feature_extractor) + 1

    classifier_input = tf.keras.Input(shape=feature_extractor.output_shape[1:])
    x = classifier_input

    for layer in model.layers[start_index:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)
    return classifier_model


def collect_images():
    image_paths = [
        path for path in SAMPLE_IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError("No valid sample images were found for Grad-CAM.")

    return sorted(image_paths)[:MAX_IMAGES_TO_PROCESS]


def load_image_for_model(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image


def load_image_for_visualization(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def make_gradcam_heatmap(model, processed_image, pred_index=None):
    feature_extractor = get_feature_extractor(model)
    classifier_model = build_classifier_head(model, feature_extractor)

    with tf.GradientTape() as tape:
        feature_maps = feature_extractor(processed_image, training=False)
        tape.watch(feature_maps)

        predictions = classifier_model(feature_maps, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_visuals(original_image, overlay_image, output_prefix):
    original_path = FIGURES_DIR / f"{output_prefix}_original.png"
    gradcam_path = FIGURES_DIR / f"{output_prefix}_gradcam.png"

    cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gradcam_path), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

    return original_path, gradcam_path


def run_gradcam():
    ensure_directories()
    model, class_names = load_artifacts()

    image_paths = collect_images()
    rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        processed_image = load_image_for_model(image_path)
        original_image = load_image_for_visualization(image_path)

        heatmap, predicted_index, probabilities = make_gradcam_heatmap(model, processed_image)
        overlay_image = overlay_heatmap(original_image, heatmap)

        predicted_label = class_names[predicted_index]
        predicted_confidence = float(probabilities[predicted_index])

        output_prefix = f"gradcam_{idx:02d}_{predicted_label}"
        original_path, gradcam_path = save_visuals(original_image, overlay_image, output_prefix)

        row = {
            "image_path": image_path.as_posix(),
            "file_name": image_path.name,
            "true_label_from_folder": image_path.parent.name,
            "predicted_label": predicted_label,
            "predicted_confidence": round(predicted_confidence, 6),
            "original_output_path": original_path.as_posix(),
            "gradcam_output_path": gradcam_path.as_posix(),
        }

        for class_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = round(float(probabilities[class_idx]), 6)

        rows.append(row)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(GRADCAM_REPORT_PATH, index=False)
    report_df.to_csv(POWERBI_GRADCAM_REPORT_PATH, index=False)

    print("Grad-CAM generation completed successfully.")
    print(f"Grad-CAM report saved to: {GRADCAM_REPORT_PATH.as_posix()}")
    print("\nGrad-CAM preview:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    run_gradcam()
