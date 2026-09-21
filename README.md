# Iris Flower Classification using K-Nearest Neighbors (KNN)

An end-to-end Machine Learning implementation of the **K-Nearest Neighbors (KNN)** algorithm on Fisher's Iris Dataset using **Scikit-Learn**, **Pandas**, and **Matplotlib**.

---

## 📌 Project Overview

K-Nearest Neighbors (KNN) is an instance-based, non-parametric lazy learning algorithm. Rather than constructing a generalized model during fitting, it memorizes the training data and performs classification at runtime by measuring distance metrics (such as Euclidean distance) to the closest $k$ neighbors.

In this project, a KNN classifier is configured with $k=3$ to predict the flower species across three distinct classes:
- **Setosa** (`0`)
- **Versicolor** (`1`)
- **Virginica** (`2`)

---

## 📊 Dataset Description

The project utilizes the classic **Iris Dataset** loaded directly via Scikit-Learn:
- **Total Samples:** 150 instances (50 per class)
- **Input Features (4 numeric dimensions):**
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Variable:** 3 balanced target classes (`[0, 1, 2]`)

---

## ⚙️ Model Architecture & Methodology

1. **Feature Separation:** The feature matrix $X \in \mathbb{R}^{150 \times 4}$ and target array $y \in \mathbb{R}^{150}$ are prepared into Pandas DataFrames.
2. **Train-Test Split:** A random 80/20 train-test partition is executed:
   - **Training set:** 120 samples
   - **Testing set:** 30 samples
3. **Model Initialization:**
   - **Algorithm:** `KNeighborsClassifier`
   - **$k$ value (`n_neighbors`):** `3` (selected as an odd number to avoid tie-breaking ambiguities)
   - **Distance Metric:** Minkowski ($p=2$, standard Euclidean distance)
4. **Inference & Evaluation:** Predictions are generated on the unseen 20% test subset and compared against true labels.

---

## 🚀 Results & Performance

- **Test Set Size:** 30 samples
- **Accuracy Score:** **`96.67%`** (`~0.9667`)
