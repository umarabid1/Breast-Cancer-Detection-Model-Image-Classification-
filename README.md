# Breast Cancer Classification Project

A comprehensive project using **machine learning** and **deep learning** techniques to classify breast cancer tumors as **benign** or **malignant**. By leveraging both **histopathological images** (BreakHis dataset) and **diagnostic measurements** (Wisconsin dataset), this repository demonstrates data cleaning, feature engineering, model selection, and evaluation techniques to assist in **breast cancer diagnosis**.

## Table of Contents
1. [Introduction](#introduction-overview-objective-and-dataset-description)  
2. [Data Preprocessing](#data-preprocessing-cleaning-handling-missing-values-normalizationscaling)  
3. [Feature Engineering & Selection](#feature-engineering-new-feature-creation-and-feature-selection-techniques)  
4. [Model Selection & Training](#model-selection-justification-training-and-hyperparameter-tuning)  
5. [Model Evaluation](#model-evaluation-metrics-comparisons-and-insights)  
6. [Usage & Scripts](#usage--scripts)  
7. [Conclusion & Future Work](#conclusion-summary-of-key-findings-and-potential-future-work)  
8. [References](#references)

---

## Introduction: Overview, Objective, and Dataset Description

### Overview
Breast cancer remains a major global health concern. Accurate and efficient detection of malignancies can significantly improve patient outcomes. This project utilizes:

- **Histopathological images** (BreakHis dataset)  
- **Diagnostic measurements** (Breast Cancer Wisconsin dataset)

...to build **machine learning** and **deep learning** classification pipelines.

### Objective
The primary goal is to **develop a classification model** that accurately distinguishes between benign and malignant tumors, ensuring high predictive accuracy and robust generalization to new data.

### Datasets

1. **Breast Cancer Wisconsin (Diagnostic)**  
   - **Source**: UCI Machine Learning Repository  
   - **Size**: 569 samples, 30 predictive features, plus a binary target variable  
   - **Labels**: 0 = Benign, 1 = Malignant  

2. **BreakHis**  
   - **Source**: Kaggle  
   - **Images**: Histopathological slides categorized by **Benign (B)** or **Malignant (M)**  
   - **Magnification Factor**: e.g., 40x, 100x  
   - **Patient Slide ID**: Ensures no patient overlap between training and validation sets  

By integrating these two datasets, we cover both **structured diagnostic features** and **image-based tumor characteristics**.

---

## Data Preprocessing: Cleaning Steps, Handling Missing Values, and Normalization/Scaling

1. **Removing Duplicates**  
   - Both the Wisconsin dataset and BreakHis dataset were checked for duplicates to prevent biased or repeated entries.

2. **Dropping Irrelevant Columns**  
   - For instance, the “ID” column in Wisconsin data was removed as it added no predictive value.

3. **Handling Missing Values**  
   - Wisconsin dataset: No missing values found.  
   - BreakHis dataset: Images with incomplete metadata (e.g., missing magnification factor) were removed.

4. **Dataset Structuring**  
   - **BreakHis** images split into `train/` and `validation/` directories, ensuring no patient overlap.

5. **Normalization/Scaling**  
   - **Tabular data**: Applied Min-Max scaling (0–1) to numeric features (e.g., radius, perimeter).  
   - **Images**: Pixel intensity scaled to 0–1 for CNN training.

For detailed splitting logic (patient ID grouping, etc.), see **`split.py`**.

---

## Feature Engineering: New Feature Creation and Feature Selection Techniques

1. **Aggregated Measurements**  
   - Ratios between concavity and compactness, as well as a composite smoothness feature combining radius and texture.

2. **Feature Encoding**  
   - **Diagnosis**: (Benign → 0, Malignant → 1).  
   - **Magnification Metadata**: e.g., 40x → 1, 100x → 2.

3. **Variance Threshold & PCA**  
   - Low-variance features removed.  
   - **Principal Component Analysis (PCA)** used to reduce dimensionality while retaining ~95% variance.

These steps helped **improve clarity and accuracy**, stabilizing training accuracy ~96%.

---

## Model Selection: Justification, Training, and Hyperparameter Tuning

1. **Convolutional Neural Network (CNN)**  
   - Suited for image classification tasks (BreakHis).  
   - Learns spatial hierarchies in histopathological images.  
   - Outperformed traditional models (~96% accuracy).

2. **Training Process**  
   - **Train**: 80% of images  
   - **Validation**: 20%  
   - Images resized, normalized. 
   - Architecture: Convolution + MaxPooling → Flatten → Dense → Softmax/Sigmoid output.

3. **Hyperparameter Tuning**  
   - **Learning Rate**: Used a schedule to avoid overshooting.  
   - **Batch Size**: Tested 16, 32, and 64 to optimize memory usage vs. performance.

See **`Model.py`** or **`pt.py`** (multi-output PyTorch script) for the exact training flow.

---

## Model Evaluation: Metrics, Comparisons, and Insights

1. **Metrics**  
   - **Accuracy**: % of correctly classified instances.  
   - **Precision & Recall**: Key for measuring false positives/negatives in malignant predictions.  
   - **F1 Score**: Harmonic mean of precision and recall.

2. **CNN vs. Traditional Models**  
   - CNN leveraged spatial/hierarchical features → highest performance (96%).  
   - Traditional methods validated the importance of features like `radius_mean`, `texture_mean`, etc.

3. **Challenges**  
   - High computational resource demand.  
   - Maintaining balanced train/val splits to avoid overfitting.

For confusion matrices, ROC curves, or logs of each epoch, refer to the output generated by scripts like **`check.py`** or **`test.py`**.

---

## Usage & Scripts

Below is a brief guide to the main scripts (filenames as mentioned in the Milestone 4 report):

1. **`split.py`**  
   - **Purpose**: Organizes the BreakHis dataset into training/validation subdirectories.  
   - Ensures patient-based splitting (so one patient’s slides don’t appear in both sets).

2. **`pt.py`** or **`Model.py`** (Multi-Output Classification)  
   - **Purpose**: Trains a multi-output PyTorch model (binary + subtype classification).  
   - Adjust parameters like `num_epochs` or `batch_size` inside the script.

3. **`test.py`** (or `predict.py`)  
   - **Purpose**: Loads the trained multi-output model and performs **inference** on new images.  
   - Shows predicted benign/malignant label plus subtype.

4. **`check.py`**  
   - **Purpose**: Example script for loading a saved Keras model (`.keras`) and testing local images.  
   - Adjust paths (e.g., `G:\model\gmod.keras`) as needed.

5. **`dataset split.txt`**  
   - **Info document** explaining the logic behind how the BreakHis dataset is partitioned by subtype and patient ID.

---

## Conclusion: Summary of Key Findings and Potential Future Work

### Key Findings
- **Model Performance**: CNN achieved ~96% accuracy, validating the effectiveness of both data preprocessing and network architecture.  
- **Feature Importance**: `radius_mean`, `concavity_mean`, and `texture_mean` crucial for distinguishing tumor shapes.  
- **Validation Metrics**: Stabilized at ~96% on test data.

### Potential Future Work
1. **Collect More Data**: Larger, more diverse samples improve model generalizability.  
2. **Magnification Factor Differentiation**: Fine‐tuned models for each magnification (40x, 100x, etc.).  
3. **Multiclass Classification**: Classify specific tumor subtypes, not just binary benign/malignant.  
4. **Transfer Learning**: Pretrained CNNs (ResNet, Inception) may boost accuracy and reduce training time.

---

## References
- **Johns Hopkins Medicine**: [Types of Breast Cancer](https://pathology.jhu.edu/breast/types-of-breast-cancer/)  
- **Medscape**: [Breast Cancer Overview](https://emedicine.medscape.com/article/1947145-overview)  
- **Nikon Instruments**: [MicroscopyU](https://www.microscopyu.com/)  
- **MyPathologyReport**: [Hemangioma](https://www.mypathologyreport.ca/diagnosis-library/hemangioma/)  
- **Google Colab**: [Colab Platform](https://colab.research.google.com/)  
- **Kaggle**: [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis) / [Cell Segmentation](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation?select=Images)

---
