[final_portfolio_metrics.csv](https://github.com/user-attachments/files/26540823/final_portfolio_metrics.csv)# Galaxy Morphology Classification

An end-to-end computer vision project focused on classifying galaxy images into morphology classes using deep learning, model interpretability, and analytics-ready reporting.

## Overview

This project applies transfer learning to classify galaxy images into three morphology categories:

- spiral
- elliptical
- irregular

The project was designed as a portfolio case study to demonstrate practical skills in:

- computer vision
- deep learning
- model evaluation
- interpretable AI
- analytics-driven reporting
- portfolio presentation for data analyst and data scientist roles

---

## Project Objective

The main goal of this project is to build a lightweight and portfolio-ready galaxy classification pipeline that produces:

- a trained image classification model
- evaluation metrics and visual reports
- Grad-CAM interpretability examples
- CSV outputs ready for analytical reporting tools
- strong visual assets for GitHub and LinkedIn presentation

---

## Key Results

[Uploading ftotal_images,num_classes,train_samples,validation_samples,test_samples,final_train_accuracy,final_validation_accuracy,test_accuracy,average_precision,average_recall,average_f1_score,inference_average_confidence,sample_image_accuracy,gradcam_examples
2700,3,1890,405,405,0.7873,0.7704,0.3333,0.1111,0.3333,0.1667,0.977,0.3333,6
inal_portfolio_metrics.csv…]()


- `outputs/reports/final_portfolio_metrics.csv`
- `outputs/reports/training_summary.csv`
- `outputs/reports/evaluation_summary.csv`

### Final Metrics

- **Total images used:** `XX`
- **Number of classes:** `XX`
- **Train samples:** `XX`
- **Validation samples:** `XX`
- **Test samples:** `XX`
- **Final train accuracy:** `XX.XX`
- **Final validation accuracy:** `XX.XX`
- **Test accuracy:** `XX.XX`
- **Average precision:** `XX.XX`
- **Average recall:** `XX.XX`
- **Average F1-score:** `XX.XX`

---

## Project Visuals

### Dataset Class Distribution
![Dataset Class Distribution](outputs/figures/portfolio_class_distribution.png)

### Model Accuracy Overview
![Model Accuracy Overview](outputs/figures/portfolio_accuracy_overview.png)

### Training Accuracy Curve
![Training Accuracy Curve](outputs/figures/training_accuracy_curve.png)

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Per-Class Evaluation Metrics
![Per-Class Evaluation Metrics](outputs/figures/per_class_metrics.png)

### Grad-CAM Example — Original Image
![Grad-CAM Original](outputs/figures/gradcam_01_example_original.png)

### Grad-CAM Example — Model Attention
![Grad-CAM Heatmap](outputs/figures/gradcam_01_example_gradcam.png)

> Replace the two Grad-CAM filenames above with the exact filenames generated in your `outputs/figures/` folder.

---

## Why This Project Matters

Although this is an astronomy-based image classification problem, the project was intentionally structured to support positioning for analytics and machine learning roles.

It demonstrates the ability to:

- work with unstructured image data
- build reproducible machine learning pipelines
- train and evaluate deep learning models
- generate business-friendly analytical outputs
- connect technical results to visual storytelling and reporting

---

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow
- OpenCV
- Streamlit
- Plotly
- Power BI-ready CSV outputs

---

## Dataset Strategy

This project uses a lightweight execution strategy to avoid excessive local storage usage while still producing strong portfolio outputs.

The approach includes:

- a curated subset of galaxy images
- three morphology classes
- compact train / validation / test manifests
- transfer learning with MobileNetV2
- minimal duplication of image files
- export of compact metrics and report tables

---

## Project Pipeline

The project follows this execution flow:

1. **Label Mapping**  
   Convert original Galaxy Zoo label structures into three portfolio-friendly classes.

2. **Data Preparation**  
   Build train, validation, and test manifests and prepare sample images.

3. **Model Training**  
   Train a lightweight transfer learning classifier using MobileNetV2.

4. **Evaluation**  
   Generate classification metrics, confusion matrix, and test predictions.

5. **Inference**  
   Run batch and single-image predictions.

6. **Grad-CAM**  
   Create visual explanations showing where the model focuses.

7. **Reporting**  
   Consolidate final metrics and figures for portfolio presentation.

---

## Project Structure

```text
galaxy-morphology-classification/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample_images/
├── notebooks/
│   └── project_walkthrough.md
├── src/
│   ├── label_mapping.py
│   ├── data_preparation.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── gradcam.py
│   ├── reporting.py
│   └── chart_style.py
├── models/
├── outputs/
│   ├── figures/
│   ├── reports/
│   ├── predictions/
│   └── powerbi/
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Main Outputs

### Reports
- classification report
- evaluation summary
- inference summary
- Grad-CAM summary
- final portfolio metrics

### Figures
- class distribution
- training curves
- confusion matrix
- per-class metrics
- Grad-CAM examples
- portfolio summary figures

### Predictions
- test predictions
- inference results

### Power BI Assets
- CSV files prepared for reporting and dashboard creation

---

## How to Run

Create the virtual environment and install dependencies:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run the full pipeline in this order:

```bash
.\.venv\Scripts\python src\label_mapping.py
.\.venv\Scripts\python src\data_preparation.py
.\.venv\Scripts\python src\training.py
.\.venv\Scripts\python src\evaluation.py
.\.venv\Scripts\python src\inference.py
.\.venv\Scripts\python src\gradcam.py
.\.venv\Scripts\python src\reporting.py
```

---

## Portfolio Positioning

This project was built to support positioning for roles such as:

- Data Analyst
- Data Scientist
- Machine Learning Analyst
- Computer Vision Analyst
- AI-focused Analytics roles

It is especially valuable as a portfolio case because it combines:

- image classification
- transfer learning
- evaluation and model reporting
- interpretability
- analytics-ready outputs
- clean GitHub presentation

---

## Future Improvements

Possible future improvements include:

- richer Streamlit interface for live predictions
- stronger dashboard layer in Power BI
- more advanced hyperparameter tuning
- expanded interpretability examples
- support for additional morphology classes

---

## Author

**Peterson Vital**  
Mechanical Engineer | Data Analytics | Machine Learning

- LinkedIn: [linkedin.com/in/petersonvital](https://linkedin.com/in/petersonvital)

This project is part of my portfolio to showcase applied computer vision, deep learning, interpretable AI, and analytics-ready reporting for real-world classification problems.
