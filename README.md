# Galaxy Morphology Classification

An end-to-end computer vision project focused on classifying galaxy images into morphology classes using deep learning, analytics-driven evaluation, and portfolio-ready reporting.

## Overview

This project explores how deep learning can be applied to astronomical image classification in a lightweight and presentation-focused way.

The main objective is to classify galaxy images into morphology categories such as:

- spiral
- elliptical
- irregular

The project was designed as a portfolio case study to demonstrate practical skills in computer vision, deep learning, model evaluation, data storytelling, and reporting.

## Business and Portfolio Context

Although this is a scientific image classification problem, the project was intentionally structured to support positioning for data analyst and data scientist roles.

It demonstrates the ability to:

- work with unstructured image data
- build and evaluate machine learning pipelines
- communicate model results clearly
- create analytics-ready outputs for reporting tools such as Power BI
- present technical work in a business-friendly format

## Project Goals

The main goals of this project are:

- build a lightweight galaxy image classification pipeline
- classify galaxy images into three morphology classes
- use transfer learning to improve performance with a smaller dataset
- generate visual outputs for portfolio presentation
- create analytics-ready reports and CSV outputs
- prepare results for GitHub, LinkedIn, and Power BI communication

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
- Power BI

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
│   ├── data_preparation.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── gradcam.py
│   ├── reporting.py
│   └── run_pipeline.py
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

## Methodology

The project is structured in six main stages:

1. **Data Preparation**  
   Organize, filter, and prepare a lightweight galaxy image dataset for training, validation, and testing.

2. **Model Training**  
   Use transfer learning with a pre-trained convolutional neural network to classify galaxy morphology.

3. **Model Evaluation**  
   Measure performance using classification metrics, confusion matrix, and per-class analysis.

4. **Interpretability**  
   Generate visual explanations such as Grad-CAM to understand model attention.

5. **Reporting**  
   Export figures, metrics, and CSV summaries for portfolio presentation and Power BI analysis.

6. **Application Layer**  
   Prepare an optional Streamlit app for image upload and live prediction.

## Dataset Strategy

This project follows a lightweight execution strategy to avoid excessive local storage usage while still producing strong portfolio results.

The planned approach includes:

- a reduced and curated subset of galaxy images
- three clearly defined morphology classes
- image resizing to a standard format
- train / validation / test split
- minimal duplication of image files
- export of compact metrics and report tables

## Planned Outputs

The project is expected to generate:

### Figures
- class distribution chart
- sample image grid by class
- training accuracy curve
- training loss curve
- confusion matrix
- precision and recall by class
- Grad-CAM examples

### Reports
- classification report
- model summary metrics
- per-class evaluation table
- training history summary

### Predictions
- example predictions
- correct vs incorrect classifications
- probability outputs for selected samples

### Power BI Assets
- CSV tables for dashboard creation
- model metrics summary
- prediction analysis tables
- screenshot-ready dashboard outputs

## Planned Scripts

### `src/data_preparation.py`
Prepare metadata, split the dataset, and organize image references for training and evaluation.

### `src/training.py`
Train the galaxy classification model using transfer learning.

### `src/evaluation.py`
Evaluate model performance and generate metrics, reports, and charts.

### `src/inference.py`
Run predictions on sample images and export prediction outputs.

### `src/gradcam.py`
Generate Grad-CAM visual explanations for selected galaxy images.

### `src/reporting.py`
Consolidate outputs into presentation-ready figures and report tables.

### `src/run_pipeline.py`
Run the full project pipeline in sequence.

## Current Status

**In progress**

The repository structure and project planning phase are complete. The next steps are dataset preparation, training pipeline setup, evaluation outputs, and final portfolio presentation.

## Portfolio Positioning

This project was designed to support positioning for roles such as:

- Data Analyst
- Data Scientist
- Machine Learning Analyst
- Computer Vision Analyst
- Analytics professional with AI and reporting skills

It is especially valuable as a portfolio case because it combines:

- computer vision
- deep learning
- scientific data interpretation
- reporting and visualization
- and business-friendly presentation

## Next Steps

- prepare the lightweight dataset structure
- create the first data preparation script
- build the training pipeline
- generate model evaluation outputs
- create Power BI-ready reporting files
- add visual assets to the README
- prepare LinkedIn and resume materials

## Author

**Peterson Vital**  
Mechanical Engineer | Data Analytics | Machine Learning

- LinkedIn: [linkedin.com/in/petersonvital](https://linkedin.com/in/petersonvital)

---

This project is part of my portfolio to showcase applied computer vision, deep learning, analytics-driven reporting, and machine learning for real-world classification problems.
