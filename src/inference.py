from pathlib import Path
import argparse
import json

import pandas as pd
import tensorflow as tf


MODEL_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

SAMPLE_IMAGES_DIR = Path("data/sample_images")
PREDICTIONS_DIR = Path("outputs/predictions")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")

INFERENCE_RESULTS_PATH = PREDICTIONS_DIR / "inference_results.csv"
INFERENCE_SUMMARY_PATH = REPORTS_DIR / "inference_summary.csv"
POWERBI_INFERENCE_RESULTS_PATH = POWERBI_DIR / "inference_results.csv"
POWERBI_INFERENCE_SUMMARY_PATH = POWERBI_DIR / "inference_summary.csv"

IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_directories():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
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


def collect_sample_images():
    if not SAMPLE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Sample images directory not found: {SAMPLE_IMAGES_DIR.as_posix()}")

    image_paths = [
        path for path in SAMPLE_IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError("No valid sample images were found for inference.")

    return sorted(image_paths)


def preprocess_image(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image


def predict_single_image(model, class_names, image_path):
    processed_image = preprocess_image(image_path)
    probabilities = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(probabilities.argmax())
    predicted_label = class_names[predicted_index]
    predicted_confidence = float(probabilities[predicted_index])

    result = {
        "image_path": Path(image_path).as_posix(),
        "file_name": Path(image_path).name,
        "true_label_from_folder": Path(image_path).parent.name if Path(image_path).parent.parent == SAMPLE_IMAGES_DIR else None,
        "predicted_label": predicted_label,
        "predicted_confidence": round(predicted_confidence, 6),
    }

    for idx, class_name in enumerate(class_names):
        result[f"prob_{class_name}"] = round(float(probabilities[idx]), 6)

    return result


def run_batch_inference(model, class_names):
    image_paths = collect_sample_images()
    rows = [predict_single_image(model, class_names, image_path) for image_path in image_paths]

    results_df = pd.DataFrame(rows)

    if "true_label_from_folder" in results_df.columns:
        results_df["is_correct"] = (
            results_df["true_label_from_folder"] == results_df["predicted_label"]
        ).astype("int32")
    else:
        results_df["is_correct"] = None

    results_df.to_csv(INFERENCE_RESULTS_PATH, index=False)
    results_df.to_csv(POWERBI_INFERENCE_RESULTS_PATH, index=False)

    summary = {
        "total_images": int(len(results_df)),
        "average_confidence": round(float(results_df["predicted_confidence"].mean()), 4),
    }

    if results_df["is_correct"].notna().all():
        summary["correct_predictions"] = int(results_df["is_correct"].sum())
        summary["incorrect_predictions"] = int((1 - results_df["is_correct"]).sum())
        summary["accuracy_on_sample_images"] = round(float(results_df["is_correct"].mean()), 4)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(INFERENCE_SUMMARY_PATH, index=False)
    summary_df.to_csv(POWERBI_INFERENCE_SUMMARY_PATH, index=False)

    return results_df, summary_df


def run_single_inference(model, class_names, image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.as_posix()}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    row = predict_single_image(model, class_names, image_path)
    results_df = pd.DataFrame([row])

    results_df.to_csv(INFERENCE_RESULTS_PATH, index=False)
    results_df.to_csv(POWERBI_INFERENCE_RESULTS_PATH, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "total_images": 1,
                "average_confidence": round(float(results_df["predicted_confidence"].mean()), 4),
            }
        ]
    )

    summary_df.to_csv(INFERENCE_SUMMARY_PATH, index=False)
    summary_df.to_csv(POWERBI_INFERENCE_SUMMARY_PATH, index=False)

    return results_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on galaxy images.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a single image for prediction.",
    )
    return parser.parse_args()


def main():
    ensure_directories()
    args = parse_args()

    model, class_names = load_artifacts()

    if args.image:
        results_df, summary_df = run_single_inference(model, class_names, args.image)
        print("Single-image inference completed successfully.")
    else:
        results_df, summary_df = run_batch_inference(model, class_names)
        print("Batch inference completed successfully.")

    print(f"Inference results saved to: {INFERENCE_RESULTS_PATH.as_posix()}")
    print(f"Inference summary saved to: {INFERENCE_SUMMARY_PATH.as_posix()}")

    print("\nInference summary:")
    print(summary_df.to_string(index=False))

    print("\nInference preview:")
    print(results_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
