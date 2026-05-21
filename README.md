# Customer Purchase Prediction Using Machine Learning

## Project Overview
This project focuses on predicting whether a customer will purchase a product based on customer details such as age, gender, city, and monthly income using Machine Learning algorithms.

The project demonstrates the complete machine learning workflow including:
- Data preprocessing
- Handling missing values
- Encoding categorical data
- Training machine learning models
- Evaluating model performance
- Visualizing results using a confusion matrix

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Dataset Information

| Column Name | Description |
|---|---|
| Customer_ID | Unique customer ID |
| Age | Customer age |
| Gender | Male/Female |
| City | Customer city |
| Monthly_Income | Monthly income |
| Purchased_Product | Target variable (0 = No, 1 = Yes) |

---

## Machine Learning Algorithms Used
- Decision Tree Classifier
- Random Forest Classifier

---

## Features of the Project
- Handles missing values using mean imputation
- Converts categorical values into numerical format using Label Encoding
- Splits dataset into training and testing sets
- Trains and tests machine learning models
- Calculates accuracy score
- Generates confusion matrix
- Displays classification report

---

## Project Workflow
1. Import Libraries
2. Load Dataset
3. Handle Missing Values
4. Encode Categorical Data
5. Split Dataset
6. Train Machine Learning Models
7. Predict Results
8. Evaluate Model Performance
9. Visualize Confusion Matrix

---

## Expected Output
The project produces:
- Dataset preview
- Missing value analysis
- Decision Tree accuracy
- Random Forest accuracy
- Confusion Matrix visualization
- Classification Report

---

## How to Run the Project

### Step 1
Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
