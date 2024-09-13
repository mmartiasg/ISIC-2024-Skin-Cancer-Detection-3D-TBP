# ISIC-2024-Skin-Cancer-Detection-3D-TBP
- ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-1.21-blue?logo=numpy&logoColor=white)

## Table of Contents
- [Description](#Description)
- [Evaluation](#Evalutaion)
- [Dataset](#Dataset)
- [Approach](#Aproach)
- [License](#license)

# Description

Skin cancer can be deadly if not caught early, but many populations lack specialized dermatologic care. Over the past several years, dermoscopy-based AI algorithms have been shown to benefit clinicians in diagnosing melanoma, basal cell, and squamous cell carcinoma. However, determining which individuals should see a clinician in the first place has great potential impact. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

Dermatoscope images reveal morphologic features not visible to the naked eye, but these images are typically only captured in dermatology clinics. Algorithms that benefit people in primary care or non-clinical settings must be adept to evaluating lower quality images. This competition leverages 3D TBP to present a novel dataset of every single lesion from thousands of patients across three continents with images resembling cell phone photos.

This competition challenges you to develop AI algorithms that differentiate histologically-confirmed malignant skin lesions from benign lesions on a patient. Your work will help to improve early diagnosis and disease prognosis by extending the benefits of automated skin cancer detection to a broader population and settings.

# Evalutaion
Submissions are evaluated on partial area under the ROC curve (pAUC) above 80% true positive rate (TPR) for binary classification of malignant examples. (See the implementation in the notebook ISIC pAUC-aboveTPR.)

[Competition link](https://www.kaggle.com/competitions/isic-2024-challenge)

# Dataset
This dataset is highly imbalance with less than 1% for the anomalous class. 400,000 RGB images are provided with variable size (no more than 128x128) and centered. There is also metadata aviable for this competition. 

# Solution proposal

## Deal with the imbalance
For this aspect I'm using a process to generate several images slightly modified with rotations, translations and noised added with a configuration to allow a certain balance it is possible to chose a 45/65 or 50/50, the more close the ratio the more syntech images are going to be generated.

## Model
An hybrid solution combining a Vitb16 transformer with an attention mechanism on certain hand-crafted features were the most promissing approach.

## Discarted approaches
Some others approaches rejected were tested such as using otsu for a threshold segmentation, applying Gabor filters on top of that, LBP with R=1 and P=8, Hog with segmentation and on the whole image. On top of that a PCA to compress the information.
