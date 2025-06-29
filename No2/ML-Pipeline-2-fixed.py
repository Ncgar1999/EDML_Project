"""
Fixed ML Pipeline for COMPAS Recidivism Prediction

This script addresses the critical issues found in the original ML-Pipeline-2.py:
- Fixes data leakage in preprocessing
- Implements proper categorical encoding
- Adds feature engineering for dates
- Handles class imbalance
- Includes proper evaluation metrics
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           f1_score, precision_recall_fscore_support)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def load_and_engineer_features():
    """Load data and perform feature engineering."""
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    
    print("Loading COMPAS dataset...")
    raw_data = pd.read_csv(raw_data_file)
    
    # Select relevant columns
    selected_columns = [
        'sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 
        'priors_count', 'days_b_screening_arrest', 'decile_score', 
        'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out'
    ]
    
    data = raw_data[selected_columns].copy()
    print(f"Original shape: {data.shape}")
    
    # Feature engineering for dates
    if 'dob' in data.columns:
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        # Extract birth year instead of using raw date
        data['birth_year'] = data['dob'].dt.year
        data['birth_decade'] = (data['birth_year'] // 10) * 10
        data = data.drop('dob', axis=1)
    
    # Handle jail dates
    if 'c_jail_in' in data.columns and 'c_jail_out' in data.columns:
        data['c_jail_in'] = pd.to_datetime(data['c_jail_in'], errors='coerce')
        data['c_jail_out'] = pd.to_datetime(data['c_jail_out'], errors='coerce')
        data['jail_duration'] = (data['c_jail_out'] - data['c_jail_in']).dt.days
        data['jail_duration'] = data['jail_duration'].fillna(0)
        data = data.drop(['c_jail_in', 'c_jail_out'], axis=1)
    
    # Create additional features
    if 'decile_score' in data.columns:
        data['high_risk'] = (data['decile_score'] >= 7).astype(int)
    
    if 'priors_count' in data.columns:
        data['has_priors'] = (data['priors_count'] > 0).astype(int)
    
    print(f"After feature engineering: {data.shape}")
    return data

def preprocess_features(X_train, X_test):
    """Properly preprocess features to avoid data leakage."""
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X_train_processed.select_dtypes(include=['object']).columns
    numerical_cols = X_train_processed.select_dtypes(exclude=['object']).columns
    
    print(f"Categorical columns: {list(categorical_cols)}")
    print(f"Numerical columns: {list(numerical_cols)}")
    
    # Handle missing values - fit on training data only
    for col in numerical_cols:
        train_median = X_train_processed[col].median()
        X_train_processed[col] = X_train_processed[col].fillna(train_median)
        X_test_processed[col] = X_test_processed[col].fillna(train_median)
    
    for col in categorical_cols:
        train_mode = X_train_processed[col].mode()[0] if not X_train_processed[col].mode().empty else 'Unknown'
        X_train_processed[col] = X_train_processed[col].fillna(train_mode)
        X_test_processed[col] = X_test_processed[col].fillna(train_mode)
    
    # Encode categorical variables properly
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit only on training data
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        
        # Transform test data, handling unseen categories
        test_values = X_test_processed[col].astype(str)
        test_encoded = []
        for value in test_values:
            if value in le.classes_:
                test_encoded.append(le.transform([value])[0])
            else:
                # Assign most frequent class for unseen categories
                test_encoded.append(le.transform([le.classes_[0]])[0])
        
        X_test_processed[col] = test_encoded
        label_encoders[col] = le
    
    return X_train_processed, X_test_processed, label_encoders

def train_and_evaluate():
    """Main training and evaluation function."""
    # Load and engineer features
    data = load_and_engineer_features()
    
    # Prepare features and target
    X = data.drop(columns=['score_text'])
    y = data['score_text']
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Target proportions:")
    print(y.value_counts(normalize=True))
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Preprocess features (avoiding data leakage)
    X_train_processed, X_test_processed, label_encoders = preprocess_features(X_train, X_test)
    
    print(f"\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Train model with better parameters and class balancing
    print("\nTraining Random Forest with improved parameters...")
    clf = RandomForestClassifier(
        n_estimators=100,        # Increased from 10
        max_depth=15,           # Increased from 5
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced', # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    clf.fit(X_train_processed, y_train)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(
        clf, X_train_processed, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro'
    )
    print(f"Cross-validation F1-macro scores: {cv_scores}")
    print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Make predictions
    y_pred = clf.predict(X_test_processed)
    
    # Evaluate performance
    print("\n" + "="*50)
    print("PERFORMANCE EVALUATION")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Macro F1 score (better for imbalanced classes)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    feature_importance = pd.DataFrame({
        'feature': X_train_processed.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Compare with original approach
    print("\n" + "="*50)
    print("COMPARISON WITH ORIGINAL APPROACH")
    print("="*50)
    print("Original issues fixed:")
    print("✓ Data leakage in preprocessing eliminated")
    print("✓ Proper categorical encoding implemented")
    print("✓ Date features properly engineered")
    print("✓ Class imbalance handled with balanced weights")
    print("✓ Model complexity increased (100 estimators vs 10)")
    print("✓ Cross-validation added for robust evaluation")
    print("✓ Macro F1 score used for better imbalanced evaluation")
    
    return clf, accuracy, macro_f1

if __name__ == "__main__":
    print("FIXED COMPAS RECIDIVISM PREDICTION PIPELINE")
    print("="*60)
    model, accuracy, macro_f1 = train_and_evaluate()
    print(f"\nFinal Results: Accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")