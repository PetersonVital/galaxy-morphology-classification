from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


TRAIN_MANIFEST_PATH = Path("data/processed/train_manifest.csv")
VALIDATION_MANIFEST_PATH = Path("data/processed/validation_manifest.csv")

MODELS_DIR = Path("models")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
POWERBI_DIR = Path("outputs/powerbi")

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.csv"
POWERBI_HISTORY_PATH = POWERBI_DIR / "training_history.csv"
POWERBI_SUMMARY_PATH = POWERBI_DIR / "training_summary.csv"

ACCURACY_FIGURE_PATH = FIGURES_DIR / "training_accuracy_curve.png"
LOSS_FIGURE_PATH = FIGURES_DIR / "training_loss_curve.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 6
FINE_TUNE_EPOCHS = 3
FINE_TUNE_AT = 120
RANDOM_SEED = 42


def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)


def load_manifests():
    if not TRAIN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Train manifest not found: {TRAIN_MANIFEST_PATH.as_posix()}")

    if not VALIDATION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {VALIDATION_MANIFEST_PATH.as_posix()}"
        )

    train_df = pd.read_csv(TRAIN_MANIFEST_PATH)
    val_df = pd.read_csv(VALIDATION_MANIFEST_PATH)

    required_columns = {"file_path", "label", "image_id"}
    for name, df in {"train": train_df, "validation": val_df}.items():
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {name} manifest: {missing}")

    return train_df, val_df


def create_label_mapping(train_df, val_df):
    class_names = sorted(set(train_df["label"].unique()) | set(val_df["label"].unique()))
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    mapping_df = pd.DataFrame(
        [{"label": label, "class_index": idx} for label, idx in label_to_index.items()]
    )
    mapping_df.to_csv(REPORTS_DIR / "class_index_mapping.csv", index=False)
    mapping_df.to_csv(POWERBI_DIR / "class_index_mapping.csv", index=False)

    return class_names, label_to_index


def decode_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(df, label_to_index, training=False):
    file_paths = df["file_path"].astype(str).tolist()
    labels = df["label"].map(label_to_index).astype("int32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=RANDOM_SEED)

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def create_callbacks():
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH.as_posix(),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    return callbacks


def fine_tune_model(model, base_model):
    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-5)


def combine_history(history_phase_1, history_phase_2=None):
    history_df_1 = pd.DataFrame(history_phase_1.history)
    history_df_1["epoch"] = range(1, len(history_df_1) + 1)
    history_df_1["phase"] = "feature_extraction"

    if history_phase_2 is None:
        return history_df_1

    history_df_2 = pd.DataFrame(history_phase_2.history)
    start_epoch = len(history_df_1) + 1
    history_df_2["epoch"] = range(start_epoch, start_epoch + len(history_df_2))
    history_df_2["phase"] = "fine_tuning"

    full_history_df = pd.concat([history_df_1, history_df_2], ignore_index=True)
    return full_history_df


def plot_training_accuracy(history_df):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(history_df["epoch"], history_df["accuracy"], linewidth=2, label="Train Accuracy")
    ax.plot(history_df["epoch"], history_df["val_accuracy"], linewidth=2, label="Validation Accuracy")

    ax.set_title("Training and Validation Accuracy", loc="left", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.savefig(ACCURACY_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(history_df):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(history_df["epoch"], history_df["loss"], linewidth=2, label="Train Loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], linewidth=2, label="Validation Loss")

    ax.set_title("Training and Validation Loss", loc="left", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.savefig(LOSS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_training_summary(history_df, class_names, train_df, val_df):
    last_row = history_df.iloc[-1]

    summary_df = pd.DataFrame(
        [
            {
                "image_height": IMAGE_SIZE[0],
                "image_width": IMAGE_SIZE[1],
                "batch_size": BATCH_SIZE,
                "initial_epochs": INITIAL_EPOCHS,
                "fine_tune_epochs": FINE_TUNE_EPOCHS,
                "total_epochs_recorded": int(history_df["epoch"].max()),
                "num_classes": len(class_names),
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "final_train_accuracy": round(float(last_row["accuracy"]), 4),
                "final_val_accuracy": round(float(last_row["val_accuracy"]), 4),
                "final_train_loss": round(float(last_row["loss"]), 4),
                "final_val_loss": round(float(last_row["val_loss"]), 4),
            }
        ]
    )

    summary_df.to_csv(TRAINING_SUMMARY_PATH, index=False)
    summary_df.to_csv(POWERBI_SUMMARY_PATH, index=False)

    return summary_df


def run_training():
    ensure_directories()

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    train_df, val_df = load_manifests()
    class_names, label_to_index = create_label_mapping(train_df, val_df)

    train_dataset = build_dataset(train_df, label_to_index, training=True)
    val_dataset = build_dataset(val_df, label_to_index, training=False)

    model, base_model = build_model(num_classes=len(class_names))
    compile_model(model, learning_rate=1e-3)

    callbacks = create_callbacks()

    history_phase_1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    fine_tune_model(model, base_model)

    history_phase_2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_phase_1.epoch[-1] + 1,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = combine_history(history_phase_1, history_phase_2)
    history_df.to_csv(TRAINING_HISTORY_PATH, index=False)
    history_df.to_csv(POWERBI_HISTORY_PATH, index=False)

    plot_training_accuracy(history_df)
    plot_training_loss(history_df)

    summary_df = create_training_summary(history_df, class_names, train_df, val_df)

    print("Training completed successfully.")
    print(f"Best model saved to: {BEST_MODEL_PATH.as_posix()}")
    print(f"Class names saved to: {CLASS_NAMES_PATH.as_posix()}")
    print(f"Training history saved to: {TRAINING_HISTORY_PATH.as_posix()}")
    print(f"Training summary saved to: {TRAINING_SUMMARY_PATH.as_posix()}")
    print(f"Accuracy figure saved to: {ACCURACY_FIGURE_PATH.as_posix()}")
    print(f"Loss figure saved to: {LOSS_FIGURE_PATH.as_posix()}")

    print("\nTraining summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_training()
