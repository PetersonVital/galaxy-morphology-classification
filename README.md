# Galaxy Morphology Classification

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


## Project Objective

The main goal of this project is to build a lightweight and portfolio-ready galaxy classification pipeline that produces:

- a trained image classification model
- evaluation metrics and visual reports
- Grad-CAM interpretability examples
- CSV outputs ready for analytical reporting tools
- strong visual assets for GitHub and LinkedIn presentation


### Final Metrics

- **Total images used:** `2700`
- **Number of classes:** `3`
- **Train samples:** `1890`
- **Validation samples:** `405`
- **Test samples:** `405`
- **Final train accuracy:** `0.79`
- **Final validation accuracy:** `0.77`
- **Test accuracy:** `0.33`
- **Average precision:** `0.11`
- **Average recall:** `0.33`
- **Average F1-score:** `0.17`



## Project Visuals

### Dataset Class Distribution
<img width="1460" height="920" alt="portfolio_class_distribution" src="https://github.com/user-attachments/assets/6e69fba0-f4ff-4b2a-aa5e-da80322a32ad" />

### Model Accuracy Overview
<img width="1460" height="920" alt="portfolio_accuracy_overview" src="https://github.com/user-attachments/assets/087999e6-be26-40c8-a241-88e8214752c1" />

### Training Accuracy Curve
<img width="1820" height="1100" alt="training_accuracy_curve" src="https://github.com/user-attachments/assets/9810b03a-2983-4d7d-b0b0-1097f3bfb26b" />

### Confusion Matrix
<img width="1459" height="1100" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d37de93b-23eb-4176-ab17-9b4867d3f2fb" />

### Grad-CAM Example — Original Image
<img width="424" height="424" alt="gradcam_01_spiral_original" src="https://github.com/user-attachments/assets/b35f0d4a-00d2-4490-b20e-6132ca235e0f" />

### Grad-CAM Example — Model Attention
<img width="424" height="424" alt="gradcam_01_spiral_gradcam" src="https://github.com/user-attachments/assets/5c6f33b9-fa51-47d2-a304-80908d5c52d0" />



## Why This Project Matters

Although this is an astronomy-based image classification problem, the project was intentionally structured to support positioning for analytics and machine learning roles.

It demonstrates the ability to:

- work with unstructured image data
- build reproducible machine learning pipelines
- train and evaluate deep learning models
- generate business-friendly analytical outputs
- connect technical results to visual storytelling and reporting


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


## Dataset Strategy

This project uses a lightweight execution strategy to avoid excessive local storage usage while still producing strong portfolio outputs.

The approach includes:

- a curated subset of galaxy images
- three morphology classes
- compact train / validation / test manifests
- transfer learning with MobileNetV2
- minimal duplication of image files
- export of compact metrics and report tables


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

#### PS: The model achieved baseline-level accuracy (~0.33), indicating limited learning under lightweight training constraints. This reflects the trade-off between computational efficiency and model performance in portfolio-oriented pipelines.

---

## Author

**Peterson Vital**  
Mechanical Engineer | Data Analytics | Machine Learning

- LinkedIn: [linkedin.com/in/petersonvital](https://linkedin.com/in/petersonvital)

This project is part of my portfolio to showcase applied computer vision, deep learning, interpretable AI, and analytics-ready reporting for real-world classification problems.
