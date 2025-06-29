"""
Improved ML Pipeline for COMPAS Recidivism Prediction

This script addresses the issues found in the original ML-Pipeline-2.py:
- Fixes data leakage in preprocessing
- Implements proper categorical encoding
- Adds feature engineering for dates
- Handles class imbalance
- Includes cross-validation and hyperparameter tuning
- Adds bias analysis for fairness
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           f1_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def load_and_preprocess_data():
    """Load and perform initial preprocessing of COMPAS data."""
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    
    print("Loading COMPAS dataset...")
    raw_data = pd.read_csv(raw_data_file)
    print(f"Original dataset shape: {raw_data.shape}")
    
    # Select relevant columns
    selected_columns = [
        'sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 
        'priors_count', 'days_b_screening_arrest', 'decile_score', 
        'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out'
    ]
    
    data = raw_data[selected_columns].copy()
    print(f"Selected columns shape: {data.shape}")
    
    # Remove rows with missing target variable
    data = data.dropna(subset=['score_text'])
    print(f"After removing missing targets: {data.shape}")
    
    return data

def engineer_features(data):
    """Create meaningful features from raw data."""
    data = data.copy()
    
    # Convert date columns to datetime
    date_columns = ['dob', 'c_jail_in', 'c_jail_out']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Feature engineering for dates
    if 'dob' in data.columns:
        # Calculate age from date of birth (more accurate than provided age)
        current_date = pd.to_datetime('2016-01-01')  # Approximate dataset collection date
        data['calculated_age'] = (current_date - data['dob']).dt.days / 365.25
        
        # Extract birth year and decade
        data['birth_year'] = data['dob'].dt.year
        data['birth_decade'] = (data['birth_year'] // 10) * 10
        
        # Drop original dob column
        data = data.drop('dob', axis=1)
    
    # Jail time features
    if 'c_jail_in' in data.columns and 'c_jail_out' in data.columns:
        data['jail_duration'] = (data['c_jail_out'] - data['c_jail_in']).dt.days
        data['jail_duration'] = data['jail_duration'].fillna(0)  # No jail time if missing
        
        # Drop original jail date columns
        data = data.drop(['c_jail_in', 'c_jail_out'], axis=1)
    
    # Create risk level binary features
    if 'decile_score' in data.columns:
        data['high_risk_score'] = (data['decile_score'] >= 7).astype(int)
        data['medium_risk_score'] = ((data['decile_score'] >= 4) & (data['decile_score'] <= 6)).astype(int)
    
    # Priors count categories
    if 'priors_count' in data.columns:
        data['has_priors'] = (data['priors_count'] > 0).astype(int)
        data['multiple_priors'] = (data['priors_count'] > 1).astype(int)
        data['many_priors'] = (data['priors_count'] > 5).astype(int)
    
    print(f"After feature engineering: {data.shape}")
    return data

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline that prevents data leakage."""
    
    # Define categorical and numerical columns
    categorical_features = ['sex', 'c_charge_degree', 'race']
    numerical_features = ['age', 'priors_count', 'days_b_screening_arrest', 'decile_score', 
                         'is_recid', 'two_year_recid', 'calculated_age', 'birth_year', 
                         'birth_decade', 'jail_duration', 'high_risk_score', 'medium_risk_score',
                         'has_priors', 'multiple_priors', 'many_priors']
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Note: We'll use LabelEncoder in the main pipeline for tree-based models
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor, categorical_features, numerical_features

def encode_categorical_features(X_train, X_test, categorical_features):
    """Properly encode categorical features to prevent column mismatch."""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in X_train_encoded.columns:
            le = LabelEncoder()
            
            # Fit on training data only
            X_train_encoded[feature] = le.fit_transform(X_train_encoded[feature].astype(str))
            
            # Transform test data, handling unseen categories
            test_values = X_test_encoded[feature].astype(str)
            test_encoded = []
            
            for value in test_values:
                if value in le.classes_:
                    test_encoded.append(le.transform([value])[0])
                else:
                    # Assign a default value for unseen categories
                    test_encoded.append(-1)
            
            X_test_encoded[feature] = test_encoded
            label_encoders[feature] = le
    
    return X_train_encoded, X_test_encoded, label_encoders

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the Random Forest model with proper configuration."""
    
    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print("Class distribution in training set:")
    print(pd.Series(y_train).value_counts())
    print(f"Computed class weights: {class_weight_dict}")
    
    # Define the model with better parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Increased from 10
        max_depth=10,      # Increased from 5
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf_model, 
        param_grid, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',  # Better metric for imbalanced classes
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Use the best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        best_model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro'
    )
    
    print(f"Cross-validation F1-macro scores: {cv_scores}")
    print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Final evaluation on test set
    y_pred = best_model.predict(X_test)
    
    return best_model, y_pred, grid_search.best_params_

def evaluate_performance(y_test, y_pred):
    """Comprehensive evaluation of model performance."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    classes = np.unique(y_test)
    
    print("\nPer-class detailed metrics:")
    for i, cls in enumerate(classes):
        print(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    # Macro-averaged F1 score (better for imbalanced classes)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nMacro-averaged F1 Score: {macro_f1:.4f}")

def analyze_bias(data, y_test, y_pred):
    """Analyze potential bias in model predictions."""
    print("\n" + "="*50)
    print("BIAS ANALYSIS")
    print("="*50)
    
    # Get test indices to align with predictions
    test_data = data.iloc[y_test.index] if hasattr(y_test, 'index') else data
    
    # Analyze by race
    if 'race' in test_data.columns:
        print("\nPrediction accuracy by race:")
        for race in test_data['race'].unique():
            mask = test_data['race'] == race
            if mask.sum() > 0:
                race_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                print(f"{race}: {race_accuracy:.4f} (n={mask.sum()})")
    
    # Analyze by sex
    if 'sex' in test_data.columns:
        print("\nPrediction accuracy by sex:")
        for sex in test_data['sex'].unique():
            mask = test_data['sex'] == sex
            if mask.sum() > 0:
                sex_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                print(f"{sex}: {sex_accuracy:.4f} (n={mask.sum()})")

def main():
    """Main execution function."""
    print("IMPROVED COMPAS RECIDIVISM PREDICTION PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Engineer features
    data = engineer_features(data)
    
    # Prepare features and target
    X = data.drop(columns=['score_text'])
    y = data['score_text']
    
    print(f"\nFinal feature set shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor, categorical_features, numerical_features = create_preprocessing_pipeline()
    
    # Apply preprocessing
    print("\nApplying preprocessing pipeline...")
    
    # Filter features that actually exist in the data
    existing_categorical = [f for f in categorical_features if f in X_train.columns]
    existing_numerical = [f for f in numerical_features if f in X_train.columns]
    
    print(f"Existing categorical features: {existing_categorical}")
    print(f"Existing numerical features: {existing_numerical}")
    
    # Simple preprocessing for this example
    # Fill missing values
    for col in existing_numerical:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())  # Use training median
    
    for col in existing_categorical:
        mode_value = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
        X_train[col] = X_train[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)
    
    # Encode categorical features properly
    X_train_encoded, X_test_encoded, label_encoders = encode_categorical_features(
        X_train, X_test, existing_categorical
    )
    
    # Train and evaluate model
    model, y_pred, best_params = train_and_evaluate_model(
        X_train_encoded, X_test_encoded, y_train, y_test
    )
    
    # Evaluate performance
    evaluate_performance(y_test, y_pred)
    
    # Analyze bias
    analyze_bias(data, y_test, y_pred)
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    feature_importance = pd.DataFrame({
        'feature': X_train_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()