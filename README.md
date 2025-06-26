# 🫁 Pneumonia Detection using Fine-Tuned VGG-19

This project applies deep learning to detect pneumonia from chest X-ray images using a fine-tuned VGG-19 convolutional neural network.

## 📊 Project Overview

- **Model Used:** VGG-19 (Transfer Learning)
- **Dataset:** 5,856 chest X-rays (4,273 pneumonia, 1,583 normal)
- **Validation Accuracy:** 93.75%
- **Test Accuracy:** 91.35%
- **Test Loss:** 0.3056

## 🧪 Techniques Used

- Fine-tuned pre-trained VGG-19 model using Keras
- Data augmentation (rotation, flipping) to improve generalization
- Binary classification with softmax output
- Evaluation with precision, recall, F1-score, and confusion matrix

## 📁 Files

- `pneumonia.ipynb`: Full Jupyter Notebook containing training, evaluation, and result visualization.

## 🛠 Dependencies

Install required libraries before running:

bash
`pip install tensorflow keras matplotlib numpy scikit-learn seaborn`

## 📈 Performance Metrics
Healthy Class:

Precision: 0.96

Recall: 0.80

F1-Score: 0.87

Pneumonia Class:

Precision: 0.89

Recall: 0.98

F1-Score: 0.93

## 📷 Visual Results
Training & validation accuracy/loss plots

Confusion matrix heatmap

Misclassified example visualizations (optional if added)

## 📌 Highlights
Achieved strong performance even with limited data

Emphasized generalizability using augmentation and regularization

Transfer learning significantly improved training speed and accuracy
