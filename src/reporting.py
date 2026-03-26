from pathlib import Path

import pandas as pd
from chart_style import create_figure, format_axis, annotate_bars, annotate_barh, save_figure


REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")
FIGURES_DIR = Path("outputs/figures")

CLASS_DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.csv"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.csv"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
INFERENCE_SUMMARY_PATH = REPORTS_DIR / "inference_summary.csv"
GRADCAM_REPORT_PATH = REPORTS_DIR / "gradcam_report.csv"

FINAL_PORTFOLIO_METRICS_PATH = REPORTS_DIR / "final_portfolio_metrics.csv"
POWERBI_FINAL_PORTFOLIO_METRICS_PATH = POWERBI_DIR / "final_portfolio_metrics.csv"

CLASS_DISTRIBUTION_FIGURE = FIGURES_DIR / "portfolio_class_distribution.png"
ACCURACY_OVERVIEW_FIGURE = FIGURES_DIR / "portfolio_accuracy_overview.png"
PER_CLASS_SUPPORT_FIGURE = FIGURES_DIR / "portfolio_per_class_support.png"
AVG_CONFIDENCE_FIGURE = FIGURES_DIR / "portfolio_avg_confidence_by_class.png"


def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.as_posix()}")
    return pd.read_csv(path)


def load_optional_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def load_data():
    class_distribution_df = load_required_csv(CLASS_DISTRIBUTION_PATH)
    training_summary_df = load_required_csv(TRAINING_SUMMARY_PATH)
    evaluation_summary_df = load_required_csv(EVALUATION_SUMMARY_PATH)
    classification_report_df = load_required_csv(CLASSIFICATION_REPORT_PATH)

    inference_summary_df = load_optional_csv(INFERENCE_SUMMARY_PATH)
    gradcam_report_df = load_optional_csv(GRADCAM_REPORT_PATH)

    return (
        class_distribution_df,
        training_summary_df,
        evaluation_summary_df,
        classification_report_df,
        inference_summary_df,
        gradcam_report_df,
    )


def plot_class_distribution(class_distribution_df: pd.DataFrame):
    plot_df = class_distribution_df.sort_values("image_count", ascending=False).copy()

    fig, ax = create_figure(figsize=(8, 5))
    ax.bar(plot_df["label"], plot_df["image_count"], width=0.62)

    format_axis(
        ax,
        title="Dataset Class Distribution",
        subtitle="Number of selected galaxy images per morphology class",
        xlabel="Galaxy Morphology Class",
        ylabel="Images",
        integer_y=True,
    )
    annotate_bars(ax, fmt="{:.0f}")

    save_figure(fig, CLASS_DISTRIBUTION_FIGURE)


def plot_accuracy_overview(training_summary_df: pd.DataFrame, evaluation_summary_df: pd.DataFrame):
    train_row = training_summary_df.iloc[0]
    eval_row = evaluation_summary_df.iloc[0]

    plot_df = pd.DataFrame(
        {
            "metric": ["Train Accuracy", "Validation Accuracy", "Test Accuracy"],
            "value": [
                float(train_row["final_train_accuracy"]),
                float(train_row["final_val_accuracy"]),
                float(eval_row["overall_accuracy"]),
            ],
        }
    )

    fig, ax = create_figure(figsize=(8, 5))
    ax.bar(plot_df["metric"], plot_df["value"], width=0.62)

    format_axis(
        ax,
        title="Model Accuracy Overview",
        subtitle="Comparison between training, validation, and test performance",
        ylabel="Accuracy",
    )
    ax.set_ylim(0, 1.05)
    annotate_bars(ax, fmt="{:.3f}")

    save_figure(fig, ACCURACY_OVERVIEW_FIGURE)


def plot_per_class_support(classification_report_df: pd.DataFrame):
    plot_df = classification_report_df.sort_values("support", ascending=True).copy()

    fig, ax = create_figure(figsize=(8, 5))
    ax.barh(plot_df["label"], plot_df["support"], height=0.65)

    format_axis(
        ax,
        title="Test Support by Class",
        subtitle="Number of test samples available for each galaxy class",
        xlabel="Support",
        ylabel="Class",
        integer_y=True,
    )
    annotate_barh(ax, fmt="{:.0f}")

    save_figure(fig, PER_CLASS_SUPPORT_FIGURE)


def plot_avg_confidence_by_class(gradcam_report_df):
    if gradcam_report_df is None or gradcam_report_df.empty:
        return

    plot_df = (
        gradcam_report_df.groupby("predicted_label", as_index=False)
        .agg(avg_confidence=("predicted_confidence", "mean"))
        .sort_values("avg_confidence", ascending=False)
    )

    fig, ax = create_figure(figsize=(8, 5))
    ax.bar(plot_df["predicted_label"], plot_df["avg_confidence"], width=0.62)

    format_axis(
        ax,
        title="Average Prediction Confidence by Class",
        subtitle="Confidence score averaged across Grad-CAM processed predictions",
        xlabel="Predicted Class",
        ylabel="Average Confidence",
    )
    ax.set_ylim(0, 1.05)
    annotate_bars(ax, fmt="{:.3f}")

    save_figure(fig, AVG_CONFIDENCE_FIGURE)


def create_final_portfolio_metrics(
    class_distribution_df: pd.DataFrame,
    training_summary_df: pd.DataFrame,
    evaluation_summary_df: pd.DataFrame,
    classification_report_df: pd.DataFrame,
    inference_summary_df,
    gradcam_report_df,
):
    train_row = training_summary_df.iloc[0]
    eval_row = evaluation_summary_df.iloc[0]

    final_metrics = {
        "total_images": int(class_distribution_df["image_count"].sum()),
        "num_classes": int(train_row["num_classes"]),
        "train_samples": int(train_row["train_samples"]),
        "validation_samples": int(train_row["validation_samples"]),
        "test_samples": int(eval_row["total_test_samples"]),
        "final_train_accuracy": round(float(train_row["final_train_accuracy"]), 4),
        "final_validation_accuracy": round(float(train_row["final_val_accuracy"]), 4),
        "test_accuracy": round(float(eval_row["overall_accuracy"]), 4),
        "average_precision": round(float(classification_report_df["precision"].mean()), 4),
        "average_recall": round(float(classification_report_df["recall"].mean()), 4),
        "average_f1_score": round(float(classification_report_df["f1_score"].mean()), 4),
    }

    if inference_summary_df is not None and not inference_summary_df.empty:
        inference_row = inference_summary_df.iloc[0]
        if "average_confidence" in inference_summary_df.columns:
            final_metrics["inference_average_confidence"] = round(
                float(inference_row["average_confidence"]), 4
            )
        if "accuracy_on_sample_images" in inference_summary_df.columns:
            final_metrics["sample_image_accuracy"] = round(
                float(inference_row["accuracy_on_sample_images"]), 4
            )

    if gradcam_report_df is not None and not gradcam_report_df.empty:
        final_metrics["gradcam_examples"] = int(len(gradcam_report_df))

    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_df.to_csv(FINAL_PORTFOLIO_METRICS_PATH, index=False)
    final_metrics_df.to_csv(POWERBI_FINAL_PORTFOLIO_METRICS_PATH, index=False)

    return final_metrics_df


def run_reporting():
    ensure_directories()

    (
        class_distribution_df,
        training_summary_df,
        evaluation_summary_df,
        classification_report_df,
        inference_summary_df,
        gradcam_report_df,
    ) = load_data()

    plot_class_distribution(class_distribution_df)
    plot_accuracy_overview(training_summary_df, evaluation_summary_df)
    plot_per_class_support(classification_report_df)
    plot_avg_confidence_by_class(gradcam_report_df)

    final_metrics_df = create_final_portfolio_metrics(
        class_distribution_df,
        training_summary_df,
        evaluation_summary_df,
        classification_report_df,
        inference_summary_df,
        gradcam_report_df,
    )

    print("Reporting completed successfully.")
    print(f"Final portfolio metrics saved to: {FINAL_PORTFOLIO_METRICS_PATH.as_posix()}")
    print(f"Class distribution figure saved to: {CLASS_DISTRIBUTION_FIGURE.as_posix()}")
    print(f"Accuracy overview figure saved to: {ACCURACY_OVERVIEW_FIGURE.as_posix()}")
    print(f"Per-class support figure saved to: {PER_CLASS_SUPPORT_FIGURE.as_posix()}")

    if gradcam_report_df is not None and not gradcam_report_df.empty:
        print(f"Average confidence figure saved to: {AVG_CONFIDENCE_FIGURE.as_posix()}")

    print("\nFinal portfolio metrics:")
    print(final_metrics_df.to_string(index=False))


if __name__ == "__main__":
    run_reporting()
