import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

print("--- Starting the script ---")

# --- 1. LOAD DATA ---
try:
    data = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please download it from Kaggle and place it in the project directory.")
    exit()

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n--- Performing EDA ---")

# Check class distribution
print("Class distribution:")
print(data['Class'].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Genuine | 1: Fraud)')
plt.savefig('class_distribution.png')
print("Class distribution plot saved as 'class_distribution.png'")

# --- 3. DATA PREPROCESSING ---
print("\n--- Preprocessing Data ---")

# RobustScaler is less prone to outliers.
scaler = RobustScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop original 'Time' and 'Amount' columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Move 'Class' column to the end
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']
data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# --- 4. TRAIN-TEST SPLIT ---
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# --- 5. HANDLE CLASS IMBALANCE WITH SMOTE ---
print("\n--- Handling class imbalance with SMOTE ---")
# IMPORTANT: Apply SMOTE only on the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Shape of training set after SMOTE:", X_train_smote.shape)
print("Class distribution in training set after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# --- 6. MODEL TRAINING ---
print("\n--- Training Logistic Regression Model ---")
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train_smote, y_train_smote)
print("Model training complete.")

# --- 7. MODEL EVALUATION ---
print("\n--- Evaluating the model ---")
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Note: High recall for class 1 is very important!

# --- 8. SAVING THE MODEL AND SCALER ---
print("\n--- Saving the model and scaler ---")
joblib.dump(model, 'logistic_model.joblib')
joblib.dump(scaler, 'robust_scaler.joblib') # Save the scaler used for Amount and Time
print("Model and scaler saved successfully!")