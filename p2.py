# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("customer_data.csv")

# Display Dataset
print("Dataset Preview:\n")
print(df.head())

# Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill missing Age with mean
age_imputer = SimpleImputer(strategy='mean')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# Fill missing Monthly Income with mean
income_imputer = SimpleImputer(strategy='mean')
df['Monthly_Income'] = income_imputer.fit_transform(df[['Monthly_Income']])

# Convert categorical columns into numeric
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])

# Drop Customer_ID
df = df.drop('Customer_ID', axis=1)

# Define Features and Target
X = df.drop('Purchased_Product', axis=1)
y = df['Purchased_Product']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Decision Tree Model
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:", dt_accuracy)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, rf_predictions)

print("\nConfusion Matrix:\n")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.show()

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, rf_predictions))