from pathlib import Path
import shutil

import pandas as pd


SOURCE_ROOT = Path("data/raw/galaxy_zoo_source")
TARGET_ROOT = Path("data/raw/galaxy_images")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")

CLASS_NAMES = ["spiral", "elliptical", "irregular"]
MAX_IMAGES_PER_CLASS = 900

LABEL_FILE_NAME = "training_solutions_rev1.csv"


def ensure_directories():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        (TARGET_ROOT / class_name).mkdir(parents=True, exist_ok=True)


def find_labels_file():
    matches = list(SOURCE_ROOT.rglob(LABEL_FILE_NAME))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {LABEL_FILE_NAME} inside {SOURCE_ROOT.as_posix()}"
        )
    return matches[0]


def find_images_directory():
    candidate_dirs = []

    for path in SOURCE_ROOT.rglob("*"):
        if path.is_dir():
            jpg_count = len(list(path.glob("*.jpg")))
            if jpg_count > 100:
                candidate_dirs.append((path, jpg_count))

    if not candidate_dirs:
        raise FileNotFoundError(
            "Could not find a training image directory containing .jpg files."
        )

    candidate_dirs.sort(key=lambda x: x[1], reverse=True)
    return candidate_dirs[0][0]


def load_labels():
    labels_path = find_labels_file()
    df = pd.read_csv(labels_path)

    if "GalaxyID" not in df.columns:
        raise ValueError("Expected column 'GalaxyID' not found in labels file.")

    required_columns = [
        "Class1.1",  # smooth
        "Class1.2",  # features/disk
        "Class4.1",  # spiral pattern (yes branch)
        "Class6.1",  # anything odd (yes branch)
        "Class7.1",  # completely round
        "Class7.2",  # in-between
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected label columns: {missing}")

    return df


def compute_class_scores(df):
    df = df.copy()

    df["rounded_score"] = df["Class7.1"] + df["Class7.2"]

    df["elliptical_score"] = (
        df["Class1.1"] * df["rounded_score"] * (1 - df["Class6.1"])
    )

    df["spiral_score"] = (
        df["Class1.2"] * df["Class4.1"] * (1 - 0.35 * df["Class6.1"])
    )

    df["irregular_score"] = df["Class6.1"]

    return df


def assign_portfolio_label(row):
    elliptical_ok = (
        row["Class1.1"] >= 0.80
        and row["rounded_score"] >= 0.55
        and row["Class6.1"] <= 0.35
    )

    spiral_ok = (
        row["Class1.2"] >= 0.70
        and row["Class4.1"] >= 0.50
    )

    irregular_ok = row["Class6.1"] >= 0.65

    candidates = []

    if elliptical_ok:
        candidates.append(("elliptical", row["elliptical_score"]))

    if spiral_ok:
        candidates.append(("spiral", row["spiral_score"]))

    if irregular_ok:
        candidates.append(("irregular", row["irregular_score"]))

    if not candidates:
        return pd.Series(
            {
                "portfolio_label": None,
                "portfolio_confidence": None,
            }
        )

    best_label, best_score = sorted(candidates, key=lambda x: x[1], reverse=True)[0]

    return pd.Series(
        {
            "portfolio_label": best_label,
            "portfolio_confidence": round(float(best_score), 6),
        }
    )


def build_labeled_subset(df):
    df = df.copy()
    label_info = df.apply(assign_portfolio_label, axis=1)
    df = pd.concat([df, label_info], axis=1)

    df = df[df["portfolio_label"].notna()].copy()

    df = (
        df.sort_values(
            ["portfolio_label", "portfolio_confidence"],
            ascending=[True, False],
        )
        .groupby("portfolio_label", group_keys=False)
        .head(MAX_IMAGES_PER_CLASS)
        .reset_index(drop=True)
    )

    return df


def attach_image_paths(df, images_dir):
    df = df.copy()

    df["image_file_name"] = df["GalaxyID"].astype(str) + ".jpg"
    df["source_image_path"] = df["image_file_name"].apply(
        lambda file_name: (images_dir / file_name).as_posix()
    )

    df["image_exists"] = df["source_image_path"].apply(lambda p: Path(p).exists())

    missing_count = int((~df["image_exists"]).sum())
    if missing_count > 0:
        print(f"Warning: {missing_count} selected images were not found and will be dropped.")

    df = df[df["image_exists"]].copy().reset_index(drop=True)

    return df


def copy_selected_images(df):
    copied_rows = []

    for _, row in df.iterrows():
        source_path = Path(row["source_image_path"])
        target_path = TARGET_ROOT / row["portfolio_label"] / source_path.name

        if not target_path.exists():
            shutil.copy2(source_path, target_path)

        copied_rows.append(
            {
                "GalaxyID": row["GalaxyID"],
                "portfolio_label": row["portfolio_label"],
                "portfolio_confidence": row["portfolio_confidence"],
                "source_image_path": source_path.as_posix(),
                "target_image_path": target_path.as_posix(),
            }
        )

    copied_df = pd.DataFrame(copied_rows)
    return copied_df


def export_reports(df, copied_df):
    df.to_csv(REPORTS_DIR / "labeled_subset_full.csv", index=False)
    copied_df.to_csv(REPORTS_DIR / "copied_image_manifest.csv", index=False)

    df.to_csv(POWERBI_DIR / "labeled_subset_full.csv", index=False)
    copied_df.to_csv(POWERBI_DIR / "copied_image_manifest.csv", index=False)

    class_summary = (
        df.groupby("portfolio_label", as_index=False)
        .agg(
            image_count=("GalaxyID", "count"),
            avg_confidence=("portfolio_confidence", "mean"),
        )
        .sort_values("image_count", ascending=False)
    )

    class_summary["avg_confidence"] = class_summary["avg_confidence"].round(4)

    class_summary.to_csv(REPORTS_DIR / "portfolio_class_summary.csv", index=False)
    class_summary.to_csv(POWERBI_DIR / "portfolio_class_summary.csv", index=False)

    threshold_summary = pd.DataFrame(
        [
            {
                "max_images_per_class": MAX_IMAGES_PER_CLASS,
                "elliptical_rule": "Class1.1 >= 0.80 and rounded_score >= 0.55 and Class6.1 <= 0.35",
                "spiral_rule": "Class1.2 >= 0.70 and Class4.1 >= 0.50",
                "irregular_rule": "Class6.1 >= 0.65",
            }
        ]
    )

    threshold_summary.to_csv(REPORTS_DIR / "label_mapping_rules.csv", index=False)
    threshold_summary.to_csv(POWERBI_DIR / "label_mapping_rules.csv", index=False)

    return class_summary, threshold_summary


def run_label_mapping():
    ensure_directories()

    labels_df = load_labels()
    images_dir = find_images_directory()

    scored_df = compute_class_scores(labels_df)
    labeled_df = build_labeled_subset(scored_df)
    labeled_df = attach_image_paths(labeled_df, images_dir)

    copied_df = copy_selected_images(labeled_df)
    class_summary, threshold_summary = export_reports(labeled_df, copied_df)

    print("Label mapping completed successfully.")
    print(f"Labels file used: {find_labels_file().as_posix()}")
    print(f"Images directory used: {images_dir.as_posix()}")
    print(f"Target dataset created at: {TARGET_ROOT.as_posix()}")

    print("\nPortfolio class summary:")
    print(class_summary.to_string(index=False))

    print("\nRule summary:")
    print(threshold_summary.to_string(index=False))


if __name__ == "__main__":
    run_label_mapping()
