import pandas as pd
import argparse
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import zscore
from utils import *


def main(args):
    # Load the data
    data = pd.read_excel('./Dataset_Case.xlsx')

    # Convert issue date to binary based on average date
    time_df = pd.to_datetime(data['issue_d'])
    avg_date = time_df.mean()
    data['issue_d'] = time_df.apply(lambda x: 0 if x < avg_date else 1)

    # Handle missing values
    if args.drop == 0:
        data['emp_length'].fillna(data['emp_length'].median(), inplace=True)
        data['mort_acc'].fillna(data['mort_acc'].median(), inplace=True)
        data['revol_util'].fillna(data['revol_util'].median(), inplace=True)
        data['pub_rec_bankruptcies'].fillna(data['pub_rec_bankruptcies'].median(), inplace=True)
    else:
        data.dropna(inplace=True)
    
    # Encode categorical variables
    categorical_cols = ['loan_status_2', 'sub_grade', 'home_ownership', 'purpose', 'addr_state', 'verification_status']
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Handle low variance columns
    low_variance_cols = data.var()[data.var() < 0.01].index
    if args.low_var_action == 'remove':
        data.drop(columns=low_variance_cols, inplace=True)
    
    # Detect and handle outliers
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = data[numerical_cols].apply(zscore)
    if args.outliers_action == 'remove':
        data = data[(np.abs(z_scores) < 3).all(axis=1)]
    elif args.outliers_action == 'cap':
        for col in numerical_cols:
            upper_limit = data[col].quantile(0.99)
            lower_limit = data[col].quantile(0.01)
            data[col] = np.clip(data[col], lower_limit, upper_limit)
    
    # Select features
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    features = [col for col in data.columns if col not in ['loan_status', 'loan_status_2']]
    iv_scores = {}
    for feature in features:
        if feature in numerical_columns:
            data[f'{feature}_binned'] = pd.qcut(data[feature], q=10, duplicates='drop')
            woe, iv = calculate_woe_iv(data, f'{feature}_binned', 'loan_status')
        else:
            woe, iv = calculate_woe_iv(data, feature, 'loan_status')
        iv_scores[feature] = iv

    # Sort and select top features
    sorted_features = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, _ in sorted_features[:args.top_features]]

    # if fico range low and fico range high in data keep only fico range low
    if 'fico_range_low' in top_features and 'fico_range_high' in top_features:
        top_features.remove('fico_range_high')
    if "int_rate" in top_features and "sub_grade" in top_features:
        top_features.remove("int_rate")
    if "installment" in top_features and "loan_amnt" in top_features:
        top_features.remove("installment")

    # Prepare the data
    X = data[top_features]
    y = data['loan_status']
    age = data['age']  

    # Convert age to binary sensitive attribute (e.g., age below or equal to median as 0, above median as 1)
    median_age = age.median()
    data['age_binary'] = (age > median_age).astype(int)

    # Split the data
    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(X, y, data['age_binary'], test_size=0.2, random_state=args.seed)

    # Scaling
    if args.scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train and evaluate models
    results = {}
    models = {
        'log_reg': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=args.seed),
        'random_forest': RandomForestClassifier(n_estimators=150, random_state=args.seed),
        'xgboost': xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', n_estimators=150, learning_rate=0.1, max_depth=4, min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.8, scale_pos_weight=1, random_state=args.seed)
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics(y_test, y_pred, age_test)

        # Print results for each model
        print(f"\nModel: {model_name}")
        print(f"Fairness Metrics: {fairness_metrics}")

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        gini = 2 * auc_roc - 1
        ks = calculate_ks(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)

        # Store results for each model
        results[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc_roc': auc_roc,
            'gini': gini,
            'ks': ks,
            'accuracy': accuracy,
            'report': report
        }

        # Print results for each model
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Gini Coefficient: {gini:.4f}")
        print(f"KS Statistic: {ks:.4f}")
        print("\nClassification Report:")
        print(report)

        # Feature importance or coefficients
        if model_name == 'log_reg':
            # Logistic Regression feature coefficients
            feature_importance = pd.DataFrame({
                'feature': top_features,
                'importance': np.abs(model.coef_[0])
            }).sort_values(by='importance', ascending=False)
            print("\nFeature Importances (Logistic Regression Coefficients):")
        else:
            # Feature importances for Random Forest and XGBoost
            feature_importance = pd.DataFrame({
                'feature': top_features,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            print(f"\nFeature Importances ({model_name}):")

        print(feature_importance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for Credit Scoring')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale the features')
    parser.add_argument('--drop', type=bool, default=0, help='Whether to drop rows with missing values')
    parser.add_argument('--top_features', type=int, default=15, help='Number of top features to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--low_var_action', type=str, default='leave', help='Action to take for low variance columns')
    parser.add_argument('--outliers_action', type=str, default='leave', help='Action to take for outliers')
    args = parser.parse_args()

    main(args)
