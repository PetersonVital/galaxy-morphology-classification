from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


TEST_MANIFEST_PATH = Path("data/processed/test_manifest.csv")
MODEL_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")
PREDICTIONS_DIR = Path("outputs/predictions")

CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CLASSIFICATION_REPORT_TEXT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.csv"
PREDICTIONS_PATH = PREDICTIONS_DIR / "test_predictions.csv"

POWERBI_CLASSIFICATION_REPORT_PATH = POWERBI_DIR / "classification_report.csv"
POWERBI_CONFUSION_MATRIX_PATH = POWERBI_DIR / "confusion_matrix.csv"
POWERBI_TEST_PREDICTIONS_PATH = POWERBI_DIR / "test_predictions.csv"

CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"
PER_CLASS_METRICS_FIGURE_PATH = FIGURES_DIR / "per_class_metrics.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def ensure_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    if not TEST_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST_PATH.as_posix()}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.as_posix()}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH.as_posix()}")

    test_df = pd.read_csv(TEST_MANIFEST_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)

    return test_df, class_names, model


def decode_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def build_test_dataset(test_df, label_to_index):
    file_paths = test_df["file_path"].astype(str).tolist()
    labels = test_df["label"].map(label_to_index).astype("int32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset


def generate_predictions(test_df, class_names, model):
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    test_dataset = build_test_dataset(test_df, label_to_index)
    probabilities = model.predict(test_dataset, verbose=1)

    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = [index_to_label[idx] for idx in predicted_indices]

    predictions_df = test_df.copy().reset_index(drop=True)
    predictions_df["predicted_label"] = predicted_labels
    predictions_df["predicted_class_index"] = predicted_indices
    predictions_df["is_correct"] = (predictions_df["label"] == predictions_df["predicted_label"]).astype(int)

    for idx, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = probabilities[:, idx]

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    predictions_df.to_csv(POWERBI_TEST_PREDICTIONS_PATH, index=False)

    return predictions_df, label_to_index


def export_classification_report(predictions_df, class_names):
    report_dict = classification_report(
        predictions_df["label"],
        predictions_df["predicted_label"],
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for class_name in class_names:
        class_metrics = report_dict[class_name]
        report_rows.append(
            {
                "label": class_name,
                "precision": round(class_metrics["precision"], 4),
                "recall": round(class_metrics["recall"], 4),
                "f1_score": round(class_metrics["f1-score"], 4),
                "support": int(class_metrics["support"]),
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(CLASSIFICATION_REPORT_PATH, index=False)
    report_df.to_csv(POWERBI_CLASSIFICATION_REPORT_PATH, index=False)

    report_text = classification_report(
        predictions_df["label"],
        predictions_df["predicted_label"],
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )

    with open(CLASSIFICATION_REPORT_TEXT_PATH, "w", encoding="utf-8") as file:
        file.write(report_text)

    return report_df


def export_confusion_matrix(predictions_df, class_names):
    cm = confusion_matrix(
        predictions_df["label"],
        predictions_df["predicted_label"],
        labels=class_names,
    )

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)
    cm_df.to_csv(POWERBI_CONFUSION_MATRIX_PATH)

    return cm_df


def plot_confusion_matrix(cm_df):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    im = ax.imshow(cm_df.values, aspect="auto")

    ax.set_title("Confusion Matrix", loc="left", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(
                j,
                i,
                str(cm_df.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color="#1F1F1F",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(CONFUSION_MATRIX_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(report_df):
    plot_df = report_df.copy()
    metrics = ["precision", "recall", "f1_score"]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    x = range(len(plot_df))
    width = 0.22

    ax.bar([i - width for i in x], plot_df["precision"], width=width, label="Precision")
    ax.bar(x, plot_df["recall"], width=width, label="Recall")
    ax.bar([i + width for i in x], plot_df["f1_score"], width=width, label="F1 Score")

    ax.set_title("Per-Class Evaluation Metrics", loc="left", pad=12)
    ax.set_xlabel("Galaxy Morphology Class")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.savefig(PER_CLASS_METRICS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_prediction_summary(predictions_df):
    summary_df = pd.DataFrame(
        [
            {
                "total_test_samples": int(len(predictions_df)),
                "correct_predictions": int(predictions_df["is_correct"].sum()),
                "incorrect_predictions": int((1 - predictions_df["is_correct"]).sum()),
                "overall_accuracy": round(float(predictions_df["is_correct"].mean()), 4),
            }
        ]
    )

    summary_df.to_csv(REPORTS_DIR / "evaluation_summary.csv", index=False)
    summary_df.to_csv(POWERBI_DIR / "evaluation_summary.csv", index=False)

    return summary_df


def run_evaluation():
    ensure_directories()

    test_df, class_names, model = load_artifacts()
    predictions_df, label_to_index = generate_predictions(test_df, class_names, model)

    report_df = export_classification_report(predictions_df, class_names)
    cm_df = export_confusion_matrix(predictions_df, class_names)
    summary_df = export_prediction_summary(predictions_df)

    plot_confusion_matrix(cm_df)
    plot_per_class_metrics(report_df)

    print("Evaluation completed successfully.")
    print(f"Predictions saved to: {PREDICTIONS_PATH.as_posix()}")
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH.as_posix()}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH.as_posix()}")
    print(f"Evaluation summary saved to: {(REPORTS_DIR / 'evaluation_summary.csv').as_posix()}")
    print(f"Confusion matrix figure saved to: {CONFUSION_MATRIX_FIGURE_PATH.as_posix()}")
    print(f"Per-class metrics figure saved to: {PER_CLASS_METRICS_FIGURE_PATH.as_posix()}")

    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))

    print("\nPer-class report:")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
