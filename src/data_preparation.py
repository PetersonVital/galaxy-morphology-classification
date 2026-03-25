from pathlib import Path
import random
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_DIR = Path("data/raw/galaxy_images")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
SAMPLE_IMAGES_DIR = Path("data/sample_images")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")

CLASS_NAMES = ["spiral", "elliptical", "irregular"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MAX_IMAGES_PER_CLASS = 900
SAMPLE_IMAGES_PER_CLASS = 4
RANDOM_STATE = 42


def ensure_directories():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        (SAMPLE_IMAGES_DIR / class_name).mkdir(parents=True, exist_ok=True)


def validate_raw_structure():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DATA_DIR}\n"
            "Expected structure:\n"
            "data/raw/galaxy_images/spiral\n"
            "data/raw/galaxy_images/elliptical\n"
            "data/raw/galaxy_images/irregular"
        )

    missing_classes = [class_name for class_name in CLASS_NAMES if not (RAW_DATA_DIR / class_name).exists()]
    if missing_classes:
        raise FileNotFoundError(
            "Missing class folders in raw dataset: " + ", ".join(missing_classes)
        )


def collect_image_records():
    random.seed(RANDOM_STATE)
    records = []

    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        image_paths = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if len(image_paths) == 0:
            raise ValueError(f"No valid images found for class: {class_name}")

        image_paths = sorted(image_paths)

        if len(image_paths) > MAX_IMAGES_PER_CLASS:
            image_paths = random.sample(image_paths, MAX_IMAGES_PER_CLASS)

        for idx, image_path in enumerate(image_paths, start=1):
            file_size_kb = round(image_path.stat().st_size / 1024, 2)

            records.append(
                {
                    "image_id": f"{class_name}_{idx:05d}",
                    "label": class_name,
                    "file_name": image_path.name,
                    "file_extension": image_path.suffix.lower(),
                    "file_path": image_path.as_posix(),
                    "file_size_kb": file_size_kb,
                }
            )

    df = pd.DataFrame(records)
    return df


def split_dataset(df):
    if df["label"].value_counts().min() < 3:
        raise ValueError(
            "Each class must have at least 3 images to create train, validation, and test splits."
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def save_manifests(train_df, val_df, test_df):
    train_df.to_csv(PROCESSED_DIR / "train_manifest.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation_manifest.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_manifest.csv", index=False)

    full_manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_manifest.to_csv(PROCESSED_DIR / "full_manifest.csv", index=False)

    return full_manifest


def create_class_distribution(full_manifest):
    class_distribution = (
        full_manifest.groupby("label", as_index=False)
        .agg(
            image_count=("image_id", "count"),
            avg_file_size_kb=("file_size_kb", "mean"),
        )
        .sort_values("image_count", ascending=False)
    )

    class_distribution["avg_file_size_kb"] = class_distribution["avg_file_size_kb"].round(2)

    class_distribution.to_csv(REPORTS_DIR / "class_distribution.csv", index=False)
    class_distribution.to_csv(POWERBI_DIR / "class_distribution.csv", index=False)

    return class_distribution


def create_split_distribution(full_manifest):
    split_distribution = (
        full_manifest.groupby(["split", "label"], as_index=False)
        .agg(image_count=("image_id", "count"))
        .sort_values(["split", "label"])
    )

    split_distribution.to_csv(REPORTS_DIR / "split_distribution.csv", index=False)
    split_distribution.to_csv(POWERBI_DIR / "split_distribution.csv", index=False)

    return split_distribution


def create_summary_table(full_manifest):
    summary = pd.DataFrame(
        [
            {
                "total_images": int(len(full_manifest)),
                "number_of_classes": int(full_manifest["label"].nunique()),
                "train_images": int((full_manifest["split"] == "train").sum()),
                "validation_images": int((full_manifest["split"] == "validation").sum()),
                "test_images": int((full_manifest["split"] == "test").sum()),
                "avg_file_size_kb": round(full_manifest["file_size_kb"].mean(), 2),
                "max_images_per_class": MAX_IMAGES_PER_CLASS,
            }
        ]
    )

    summary.to_csv(REPORTS_DIR / "data_preparation_summary.csv", index=False)
    summary.to_csv(POWERBI_DIR / "data_preparation_summary.csv", index=False)

    return summary


def export_sample_images(full_manifest):
    sample_manifest_rows = []

    for class_name in CLASS_NAMES:
        class_subset = (
            full_manifest[full_manifest["label"] == class_name]
            .sort_values("image_id")
            .head(SAMPLE_IMAGES_PER_CLASS)
            .copy()
        )

        target_class_dir = SAMPLE_IMAGES_DIR / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for _, row in class_subset.iterrows():
            source_path = Path(row["file_path"])
            target_path = target_class_dir / source_path.name

            if not target_path.exists():
                shutil.copy2(source_path, target_path)

            sample_manifest_rows.append(
                {
                    "image_id": row["image_id"],
                    "label": row["label"],
                    "source_file_path": source_path.as_posix(),
                    "sample_file_path": target_path.as_posix(),
                }
            )

    sample_manifest = pd.DataFrame(sample_manifest_rows)
    sample_manifest.to_csv(REPORTS_DIR / "sample_image_manifest.csv", index=False)
    sample_manifest.to_csv(POWERBI_DIR / "sample_image_manifest.csv", index=False)

    return sample_manifest


def run_data_preparation():
    ensure_directories()
    validate_raw_structure()

    raw_df = collect_image_records()
    raw_df.to_csv(INTERIM_DIR / "raw_inventory.csv", index=False)

    train_df, val_df, test_df = split_dataset(raw_df)
    full_manifest = save_manifests(train_df, val_df, test_df)

    class_distribution = create_class_distribution(full_manifest)
    split_distribution = create_split_distribution(full_manifest)
    summary = create_summary_table(full_manifest)
    sample_manifest = export_sample_images(full_manifest)

    print("Data preparation completed successfully.")
    print(f"Train manifest: {(PROCESSED_DIR / 'train_manifest.csv').as_posix()}")
    print(f"Validation manifest: {(PROCESSED_DIR / 'validation_manifest.csv').as_posix()}")
    print(f"Test manifest: {(PROCESSED_DIR / 'test_manifest.csv').as_posix()}")
    print(f"Summary report: {(REPORTS_DIR / 'data_preparation_summary.csv').as_posix()}")
    print(f"Power BI class distribution: {(POWERBI_DIR / 'class_distribution.csv').as_posix()}")

    print("\nClass distribution preview:")
    print(class_distribution.to_string(index=False))

    print("\nSplit distribution preview:")
    print(split_distribution.to_string(index=False))

    print("\nSummary preview:")
    print(summary.to_string(index=False))

    print("\nSample image manifest preview:")
    print(sample_manifest.head().to_string(index=False))


if __name__ == "__main__":
    run_data_preparation()
