# 🧠 ADHD Detection from EEG Signals Using 1D CNN and XGBoost

This project leverages EEG data to classify individuals as either **ADHD** (Attention Deficit Hyperactivity Disorder) or **Control**. We employ two machine learning models: a **Convolutional Neural Network (CNN)** designed for time-series data and an **XGBoost** classifier for performance comparison. The models are trained and evaluated using signal features extracted from raw EEG data.

---

## 📁 Dataset

- **File**: `adhdata.csv`
- **Target Column**: `Class` (values: `"ADHD"` or `"Control"`)
- **Features**: Numerical values representing EEG signal amplitudes from multiple channels or time points.

### 🔍 Sample Format:
| ID | F1 | F2 | F3 | ... | Fn | Class |
|----|----|----|----|-----|----|-------|
| 1  | 0.2| 0.5| ...| ... |0.3 | ADHD  |

---

## ⚙️ Project Workflow

1. **Load and Inspect Data**
2. **Preprocess**:
   - Encode target labels (`ADHD` → `1`, `Control` → `0`)
   - Drop `ID` column
   - Normalize features using `StandardScaler`
3. **Split**:
   - 80% Training
   - 20% Testing (stratified to maintain class balance)
4. **Modeling**:
   - Train 1D CNN on reshaped input
   - Train XGBoost on original feature vector
5. **Evaluate**:
   - Accuracy
   - Confusion Matrix
   - Classification Report
6. **Visualize**:
   - EEG signal sample
   - Accuracy & Loss curves
   - Confusion matrix heatmaps

---

## 🧠 Model 1: Convolutional Neural Network (1D CNN)

### 🔧 Architecture:
- `Conv1D`: 64 filters, kernel size 1, ReLU activation
- `MaxPooling1D`: pool size 1
- `Dropout`: 0.5 (after conv and dense layers)
- `Flatten`
- `Dense`: 64 neurons, ReLU
- `Dense`: 2 neurons (softmax)

### 🧮 Compilation:
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: `accuracy`
- **Epochs**: 15
- **Batch Size**: 32

### 🔁 Training History:
- Training/Validation accuracy and loss over epochs is plotted.

---

## 🌲 Model 2: XGBoost Classifier

- Gradient boosting classifier optimized for speed and performance.
- Label encoding is used.
- Evaluated on:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 📊 Visualization & Plots

- **Class Distribution**: Counts of ADHD and Control classes
- **EEG Sample Plot**: Signal amplitude over time
- **CNN Training History**: Accuracy and loss over epochs
- **Confusion Matrix**: Heatmaps for both models

---

## 🚀 Getting Started

### ✅ Prerequisites

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
