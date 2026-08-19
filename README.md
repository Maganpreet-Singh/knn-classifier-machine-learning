# KNN Classifier — Machine Learning

A practical implementation of the **K-Nearest Neighbors (KNN) Classification** algorithm using Python and Scikit-learn. This project demonstrates the complete machine learning workflow on a breast cancer classification dataset, including data loading, exploratory analysis, preprocessing, feature scaling, model training, evaluation, and visualization.

## 📌 Project Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression. For classification, KNN predicts the class of a new observation by looking at the classes of its nearest training observations.

In this project, KNN is used to classify observations into **Malignant (M)** and **Benign (B)** classes based on numerical diagnostic features.

## 🎯 Objectives

- Understand the intuition behind KNN
- Explore and preprocess a real-world classification dataset
- Split data into training and testing sets
- Apply feature scaling using `StandardScaler`
- Train a KNN classifier using Scikit-learn
- Evaluate the model using accuracy, confusion matrix, and classification report
- Understand the effect of the `K` hyperparameter
- Build an end-to-end machine learning classification workflow

## 📂 Repository Contents

```text
knn-classifier-machine-learning/
│
├── KNN.ipynb       # Complete KNN implementation and analysis
├── data.csv        # Dataset used by the notebook
└── README.md       # Project documentation
```

The repository currently contains a Jupyter notebook, the dataset, and this documentation. The notebook imports NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn and implements the KNN workflow. fileciteturn2file0 fileciteturn3file0

## 📊 Dataset

The project uses a breast cancer diagnostic classification dataset. The target column is `diagnosis`:

- `M` — Malignant
- `B` — Benign

The dataset contains numerical measurements including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, with mean, standard-error, and worst-value variants.

## 🧠 What is K-Nearest Neighbors?

KNN is a **supervised, instance-based, lazy learning algorithm**. Rather than learning a conventional parametric model, it compares a new observation with nearby training observations and uses their labels to make a prediction.

For classification:

1. Choose a value of `K`.
2. Calculate distances from the new observation to training observations.
3. Select the `K` nearest observations.
4. Count their class labels.
5. Predict the majority class.

A common distance metric is Euclidean distance:

```text
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

## ⚖️ Why Feature Scaling Matters

KNN is distance-based. If one feature has values in the thousands while another has values between 0 and 1, the larger-scale feature can dominate the distance calculation.

The project therefore uses `StandardScaler` to standardize numerical features:

```text
z = (x - μ) / σ
```

The scaler should be fitted on training data and then used to transform the test data so information from the test set does not leak into training.

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## 📦 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

These imports are present in the current notebook. fileciteturn3file0

## 🔄 Machine Learning Workflow

```text
Load Dataset
      ↓
Inspect Data
      ↓
Clean / Preprocess Data
      ↓
Separate Features and Target
      ↓
Encode Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Train KNN
      ↓
Generate Predictions
      ↓
Evaluate Performance
      ↓
Visualize Results
```

## 🔎 Steps Covered

### 1. Import Libraries

Load the libraries required for data analysis, visualization, preprocessing, model training, and evaluation.

### 2. Load and Inspect Data

Read the CSV file into a Pandas DataFrame and inspect its structure, columns, data types, and missing values.

### 3. Clean the Data

Remove irrelevant identifier fields and empty columns where appropriate before fitting the classifier.

### 4. Separate Features and Target

```python
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
```

The feature matrix contains numerical measurements, while `diagnosis` is the classification target.

### 5. Encode the Target

Convert the categorical diagnosis labels into machine-readable values, for example:

```text
M → 1
B → 0
```

### 6. Train-Test Split

A typical split can be performed with:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training data is used to fit the model, while the test data is reserved for evaluation.

### 7. Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 8. Train the KNN Classifier

Example configuration:

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

### 9. Make Predictions

```python
y_pred = knn.predict(X_test_scaled)
```

### 10. Evaluate the Model

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

### 11. Visualize the Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()
```

## 📈 Evaluation Metrics

### Accuracy

The proportion of predictions that are correct:

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many observations predicted as a class actually belong to that class:

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual observations of a class were correctly identified:

```text
Recall = TP / (TP + FN)
```

### F1 Score

The harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion Matrix

For binary classification:

```text
                 Predicted
                 B       M
Actual B        TN      FP
Actual M        FN      TP
```

This helps identify false positives and false negatives instead of relying on accuracy alone.

## 🔧 Choosing the Value of K

The `n_neighbors` parameter controls how many neighbors vote on a prediction.

### Small K

- More sensitive to local patterns
- More sensitive to noise
- Can overfit

### Large K

- Smoother decision boundaries
- Less sensitive to individual observations
- Can underfit

A strong next step is to evaluate multiple values of `K` using cross-validation rather than selecting one arbitrarily.

## ✅ Key Learning Outcomes

By completing this project, you will understand:

- How KNN classification works
- Why KNN is a lazy learner
- Why distance matters in KNN
- Why feature scaling is important
- How to train KNN with Scikit-learn
- How to interpret accuracy and classification metrics
- How to read a confusion matrix
- How the choice of `K` affects model behavior
- How to structure an end-to-end classification workflow

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Maganpreet-Singh/knn-classifier-machine-learning.git
```

### 2. Move into the Project Directory

```bash
cd knn-classifier-machine-learning
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

```text
KNN.ipynb
```

Run the notebook cells from top to bottom.

## 📁 Project Structure

```text
KNN Classifier
│
├── Data Loading
├── Data Inspection
├── Data Cleaning
├── Feature Selection
├── Target Encoding
├── Train-Test Split
├── Feature Scaling
├── KNN Model Training
├── Prediction
├── Model Evaluation
└── Confusion Matrix Visualization
```

## 💡 Advantages of KNN

- Simple and intuitive
- Easy to implement
- No complex mathematical training procedure
- Can model non-linear decision boundaries
- Useful as a strong baseline classifier

## ⚠️ Limitations of KNN

- Prediction can be expensive on large datasets
- Sensitive to feature scaling
- Sensitive to irrelevant features
- Sensitive to noise and outliers
- Requires a sensible value of `K`
- Can struggle in high-dimensional feature spaces

## 🧪 Future Improvements

The repository can be extended with:

- Hyperparameter tuning with `GridSearchCV`
- Cross-validation
- Testing several `K` values
- Comparison of distance metrics
- ROC-AUC and precision-recall analysis
- Feature selection
- PCA for dimensionality reduction
- Model comparison with Logistic Regression, SVM, Decision Tree, Random Forest, and Naive Bayes
- Model persistence with `joblib`
- A Streamlit interface for interactive predictions

## 🔬 Example: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_neighbors": range(1, 21)
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train_scaled, y_train)

print("Best K:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)
```

## 📌 Best Practices

- Scale features before applying distance-based KNN when feature ranges differ.
- Fit preprocessing objects only on training data.
- Keep identifier columns out of the feature matrix unless they carry legitimate predictive information.
- Use validation or cross-validation to select `K`.
- Consider precision, recall, and F1 score in addition to accuracy.

## 📚 Concepts Demonstrated

```text
Supervised Learning
        ↓
Classification
        ↓
Distance-Based Learning
        ↓
K-Nearest Neighbors
        ↓
Feature Scaling
        ↓
Model Evaluation
        ↓
Confusion Matrix
```

## 👨‍💻 Author

**Maganpreet Singh**

B.Tech Computer Science & Engineering student focused on **Python, Machine Learning, Data Science, and AI**.

GitHub: [Maganpreet-Singh](https://github.com/Maganpreet-Singh)

## ⭐ Support

If you find this project useful for learning machine learning, consider giving the repository a ⭐ on GitHub.

## 📄 License

This project is intended for **educational and learning purposes**. You may adapt the code and extend the project for your own machine learning practice.
