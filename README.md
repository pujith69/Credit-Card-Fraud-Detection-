# Credit Card Fraud Detection System

A machine learning-based system for detecting fraudulent credit card transactions using Logistic Regression and SMOTE for handling class imbalance.

## Features

- **Fraud Detection Model**: Trained Logistic Regression classifier with high recall for detecting fraudulent transactions
- **Web Interface**: Streamlit-based app for real-time fraud prediction
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to address imbalanced data
- **Robust Scaling**: RobustScaler for Time and Amount features to handle outliers

## Project Structure

```
├── app.py                    # Streamlit web application
├── train_model.py            # Model training script
├── logistic_model.joblib     # Trained Logistic Regression model
├── robust_scaler.joblib      # Fitted RobustScaler for preprocessing
├── class_distribution.png    # Visualization of class distribution
└── README.md                 # This file
```

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains:
- 28 anonymized features (V1-V28) from PCA transformation
- Time: seconds elapsed between each transaction and the first transaction
- Amount: transaction amount
- Class: 1 for fraudulent, 0 for genuine

**Note**: The dataset is highly imbalanced (~99.8% genuine, ~0.2% fraudulent).

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn imblearn matplotlib seaborn joblib streamlit
```

Or create a requirements file:
```bash
pip freeze > requirements.txt
```

## Usage

### Train the Model

```bash
python train_model.py
```

This script will:
1. Load the creditcard.csv dataset
2. Perform EDA and save class distribution plot
3. Preprocess data with RobustScaler
4. Handle class imbalance using SMOTE
5. Train a Logistic Regression model
6. Evaluate and save the model + scaler

### Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser where you can:
- Input transaction details (Time, Amount)
- Enter the 28 V-features as comma-separated values
- Get instant fraud/genuine predictions with confidence scores

## Model Performance

The model is evaluated using:
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: True/False Positives and Negatives

**Important**: High recall for the fraud class (Class 1) is critical to minimize false negatives (missing fraudulent transactions).

## Technologies Used

- **scikit-learn**: Machine learning framework
- **imblearn (SMOTE)**: Handling class imbalance
- **Streamlit**: Web application framework
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model serialization

## License

MIT License
