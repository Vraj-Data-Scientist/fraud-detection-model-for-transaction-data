
# Credit Fraud Detection Model for Imbalanced Transaction Data

## Overview
This project develops a machine learning model to detect fraudulent transactions in credit card data, addressing the challenge of highly imbalanced datasets. The model leverages preprocessing techniques, undersampling (NearMiss), oversampling (SMOTE), and multiple classifiers (e.g., Logistic Regression, Random Forest) to identify fraud. A neural network is also implemented to compare performance with traditional classifiers. The project emphasizes best practices for handling imbalanced data, such as avoiding testing on resampled datasets and using appropriate evaluation metrics like F1-score, precision, recall, and AUC-ROC.

---

## Model Performance Results

The project evaluates multiple machine learning models and a neural network for detecting fraudulent credit card transactions, using both random undersampling and oversampling (SMOTE) to handle the highly imbalanced dataset (99.83% non-fraud, 0.17% fraud). Performance is assessed on the original test set (56,961 samples, including 98 fraud cases) using accuracy, precision, recall, F1-score, ROC-AUC, and average precision-recall (AP) scores. The following sections summarize the results for each approach.

### 1. Random Undersampling
Random undersampling creates a balanced 50/50 subset (492 fraud, 492 non-fraud) for training, with outliers removed from features V10, V12, and V14 to improve model accuracy. The models are trained on this subset and tested on the original test set.

#### Classifier Performance (Test Set, 190 Samples from Undersampled Data)
The following metrics are reported for the test set derived from the undersampled data (balanced subset):

| Classifier                  | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|-----------------------------|----------|------------------|----------------|------------------|
| Logistic Regression         | 94.21%   | 0.97             | 0.92           | 0.95             |
| K-Nearest Neighbors         | 91.58%   | 0.99             | 0.86           | 0.92             |
| Support Vector Classifier   | 92.63%   | 0.97             | 0.90           | 0.93             |
| Decision Tree Classifier    | 89.47%   | 0.91             | 0.90           | 0.91             |

- **Logistic Regression** outperforms others with the highest accuracy (94.21%) and a strong F1-score (0.95) for fraud detection.
- **K-Nearest Neighbors** achieves high precision (0.99) but lower recall (0.86), indicating fewer false positives but missing some fraud cases.
- **Support Vector Classifier** balances precision and recall well, with a 93% F1-score.
- **Decision Tree** has the lowest accuracy (89.47%) but maintains a decent F1-score (0.91).

#### ROC-AUC Scores (Training Set, Cross-Validation)
ROC-AUC scores from 5-fold cross-validation on the undersampled training set:

| Classifier                  | ROC-AUC Score |
|-----------------------------|---------------|
| Logistic Regression         | 0.9772        |
| Support Vector Classifier   | 0.9751        |
| K-Nearest Neighbors         | 0.9306        |
| Decision Tree Classifier    | 0.9239        |

- Logistic Regression achieves the highest ROC-AUC (0.9772), indicating excellent separation between fraud and non-fraud classes.

#### Neural Network (Undersampling)
A simple neural network (one hidden layer with 32 nodes, ReLU activation, softmax output) was trained on the undersampled data. Performance on the original test set (56,961 samples):

- **Confusion Matrix**:
  - True Negatives (Non-Fraud Correct): 53,669
  - False Positives (Non-Fraud as Fraud): 3,194
  - False Negatives (Fraud as Non-Fraud): 5
  - True Positives (Fraud Correct): 93
- **Metrics**:
  - Accuracy: ~94.37% (calculated as (53,669 + 93) / 56,961)
  - Precision (Fraud): 0.028 (93 / (93 + 3,194))
  - Recall (Fraud): 0.949 (93 / 98)
  - F1-Score (Fraud): ~0.055 (2 * (0.028 * 0.949) / (0.028 + 0.949))
- **Observation**: High recall (94.9%) but very low precision (2.8%) due to many false positives, making this model less practical for real-world use.

### 2. Oversampling (SMOTE)
SMOTE generates synthetic fraud samples to balance the training set, applied during cross-validation to avoid data leakage. The Logistic Regression model with SMOTE is tested on the original test set.

#### Logistic Regression with SMOTE (Original Test Set, 56,961 Samples)
- **Classification Report**:
  ```
              precision    recall  f1-score   support
  No Fraud    1.00       0.99      0.99     56,863
  Fraud       0.11       0.86      0.20        98
  Accuracy                           0.99     56,961
  Macro Avg   0.56       0.92      0.60     56,961
  Weighted Avg 1.00      0.99      0.99     56,961
  ```
- **Metrics**:
  - Accuracy: 98.81%
  - Precision (Fraud): 0.11
  - Recall (Fraud): 0.86
  - F1-Score (Fraud): 0.20
  - Average Precision-Recall Score: 0.75
- **Observation**: High accuracy (98.81%) is misleading due to class imbalance. The model achieves good recall (86%) but low precision (11%), indicating many false positives.

#### Cross-Validation Metrics (SMOTE, Training Set)
Average metrics from 5-fold cross-validation on the SMOTE-augmented training set:

- Accuracy: 94.26%
- Precision: 0.061
- Recall: 0.914
- F1-Score: 0.112
- ROC-AUC: Not explicitly reported for SMOTE in cross-validation, but the high recall suggests strong fraud detection capability.

#### Neural Network (SMOTE)
The same neural network architecture was trained on the SMOTE-augmented data. Performance on the original test set:

- **Confusion Matrix**:
  - True Negatives (Non-Fraud Correct): 56,841
  - False Positives (Non-Fraud as Fraud): 22
  - False Negatives (Fraud as Non-Fraud): 28
  - True Positives (Fraud Correct): 70
- **Metrics**:
  - Accuracy: ~99.76% ((56,841 + 70) / 56,961)
  - Precision (Fraud): 0.761 (70 / (70 + 22))
  - Recall (Fraud): 0.714 (70 / 98)
  - F1-Score (Fraud): ~0.737 (2 * (0.761 * 0.714) / (0.761 + 0.714))
- **Observation**: The SMOTE-trained neural network significantly improves precision (76.1%) compared to undersampling, with a reasonable recall (71.4%), making it more balanced and practical.

### 3. Comparison of Techniques
The final comparison of Logistic Regression performance on the original test set:

| Technique                | Accuracy  | Notes                                      |
|--------------------------|-----------|--------------------------------------------|
| Random Undersampling     | 94.21%    | High precision, good for balanced subset   |
| Oversampling (SMOTE)     | 98.81%    | High recall, but low precision due to imbalance |

- **Undersampling**: Better precision (0.97) and F1-score (0.95) on the balanced subset, but loses information due to reduced dataset size.
- **SMOTE**: Higher accuracy (98.81%) and recall (0.86) on the original test set, but precision (0.11) is low due to many false positives. The neural network with SMOTE achieves a better balance (precision: 0.761, recall: 0.714).

### 4. Key Observations
- **Logistic Regression** is the most effective classifier, with a cross-validation ROC-AUC of 0.9772 (undersampling) and strong performance across both techniques.
- **SMOTE** improves recall (86-91%) but struggles with precision (6-11%) on the original test set due to class imbalance. The neural network with SMOTE mitigates this, achieving a balanced F1-score (0.737).
- **Undersampling** yields high precision and F1-scores on the balanced subset but may not generalize as well due to information loss.
- **Neural Network**: The SMOTE-trained neural network outperforms the undersampled version, with fewer false positives and a more practical precision-recall trade-off.
- **Misleading Accuracy**: High accuracy (e.g., 98.81% with SMOTE) is misleading due to the imbalanced dataset. Focus on precision, recall, and F1-score for fraud detection.


---






