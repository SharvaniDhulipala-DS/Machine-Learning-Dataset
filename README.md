# Machine Learning Lab Tests: Comprehensive Implementation

This repository contains the solutions and implementations for various machine learning lab tests. Each test covers essential concepts and algorithms in machine learning, including supervised learning, unsupervised learning, regression, classification, clustering, and more.

## Table of Contents
1. [Lab Test 1](#lab-test-1)
    - [SET 1: Find-S Algorithm](#set-1-find-s-algorithm)
    - [SET 2: KNN Classification using Diabetes Dataset](#set-2-knn-classification-using-diabetes-dataset)
    - [SET 3: Linear Regression using Advertising Dataset](#set-3-linear-regression-using-advertising-dataset)
    - [SET 4: Naïve Bayes with Synthetic Dataset](#set-4-naïve-bayes-with-synthetic-dataset)
    - [SET 5: Linear Regression using Advertising Dataset](#set-5-linear-regression-using-advertising-dataset)
    - [SET 6: Linear Regression using Insurance Dataset](#set-6-linear-regression-using-insurance-dataset)
    - [SET 7: KNN Classification using Iris Dataset](#set-7-knn-classification-using-iris-dataset)
    - [SET 8: Find-S Algorithm](#set-8-find-s-algorithm)
2. [Lab Test 2](#lab-test-2)
    - [Clustering using Fruits Dataset](#clustering-using-fruits-dataset)
    - [Classification using Heart Dataset](#classification-using-heart-dataset)
    - [Drug Classification using Drug Dataset](#drug-classification-using-drug-dataset)

---

## Lab Test 1

### SET 1: Find-S Algorithm
- **Steps**:
    1. Import and read the dataset.
    2. Drop unnecessary columns.
    3. Print unique values from the dataset.
    4. Set the target variable (`Poisonous`).
    5. Set all other columns as features.
    6. Initialize the hypothesis to `'0'`.
    7. Implement the Find-S Algorithm for:
        - `Poisonous` values.
        - `Non-Poisonous` values.

---

### SET 2: KNN Classification using Diabetes Dataset
- **Steps**:
    1. Read the dataset and check for null values.
    2. Select features for training and set `Outcome` as the target variable.
    3. Split data into 80% training and 20% testing.
    4. Train the model with `N=5` using Manhattan metric.
    5. Make predictions on test data.
    6. Evaluate performance:
        - Classification Report.
        - Confusion Matrix.

---

### SET 3: Linear Regression using Advertising Dataset
- **Steps**:
    1. Read the dataset and rename columns:
        - `Sales` → `Total Sales`.
        - `Spend` → `Total Spend`.
    2. Display scatter plots for `Total Sales` vs. `Total Spend`.
    3. Compute the slope and intercept of the regression line.
    4. Calculate and display the dataset centroid.
    5. Predict values based on the centroid.
    6. Perform Linear Regression.
    7. Make predictions.

---

### SET 4: Naïve Bayes with Synthetic Dataset
- **Steps**:
    1. Generate a synthetic classification dataset.
    2. Display scatter plots for visualization.
    3. Select features for training/testing.
    4. Train a Naïve Bayes classifier.
    5. Print actual vs. predicted values.
    6. Calculate accuracy and F1 score.
    7. Display the confusion matrix.

---

### SET 5: Linear Regression using Advertising Dataset
- **Steps**:
    1. Read the dataset.
    2. Display data types and the first 10 rows.
    3. Create scatter plots for `Radio` vs. `Sales`.
    4. Perform statistical analysis and create a box plot for `Sales`.
    5. Train and test the model with an 80:20 split.
    6. Train a Linear Regression model and find coefficients.

---

### SET 6: Linear Regression using Insurance Dataset
- **Steps**:
    1. Read the dataset and display column data types.
    2. Display scatter plots for `BMI` vs. `Charges`.
    3. Perform statistical analysis and create a box plot for `Charges`.
    4. Set features (`X`) and target variable (`Y`).
    5. Split data into 70% training and 30% testing.
    6. Train a Linear Regression model and find coefficients.

---

### SET 7: KNN Classification using Iris Dataset
- **Steps**:
    1. Read the dataset and check for null values.
    2. Set independent features (`X`) and target variable (`Y`).
    3. Split data into 80% training and 20% testing.
    4. Train a KNN classifier (`N=7`) using Euclidean distance.
    5. Make predictions and evaluate performance:
        - Classification Report.
        - Confusion Matrix.

---

### SET 8: Find-S Algorithm
- **Steps**:
    1. Read the dataset.
    2. Display unique values.
    3. Set the target variable (`buy`).
    4. Set all other variables as features.
    5. Initialize the hypothesis to `'0'`.
    6. Implement the Find-S Algorithm for:
        - `buy` values.
        - `non-buy` values.

---

## Lab Test 2

### Clustering using Fruits Dataset
- **Steps**:
    1. Visualize `mass` and `width` features using scatterplots.
    2. Split data into training/testing sets.
    3. Find the optimal number of clusters using the elbow method.
    4. Plot performance scores to select the best `k`.
    5. Train a clustering model with the optimal `k`.
    6. Visualize the resulting clusters.

---

### Classification using Heart Dataset
- **Steps**:
    1. Load the dataset and check for null values.
    2. Select features and target variable.
    3. Split data into 75% training and 25% testing.
    4. Train a Decision Tree classifier.
    5. Predict the target variable for the test set.
    6. Evaluate performance:
        - Model Accuracy.
        - Classification Report.

---

### Drug Classification using Drug Dataset
- **Steps**:
    1. Load the dataset and check for null values.
    2. Select features and target variable.
    3. Split data into 70% training and 30% testing.
    4. Train a Decision Tree classifier.
    5. Predict the drug class for the test set.
    6. Evaluate performance:
        - Model Accuracy.
        - Confusion Matrix.

---

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `scipy`

## Usage
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/ml-lab-tests.git
   ```
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the desired script and run:  
   ```bash
   python script_name.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
