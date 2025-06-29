"""
Improved ML Pipeline for Adult Income Prediction
Fixes bugs and adds enhancements to the original ML-Pipeline-1.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """
    Find the project root directory by looking for specific markers.
    This replaces the missing utils.get_project_root function.
    """
    current_path = Path(__file__).resolve()
    
    # Look for project root indicators - prioritize datasets folder
    for parent in current_path.parents:
        if (parent / "datasets").exists():
            return parent
    
    # If no datasets folder found, look for .git but skip immediate parent
    for parent in current_path.parents:
        if (parent / ".git").exists() and parent != current_path.parent:
            return parent
    
    # Fallback to current file's parent directory
    return current_path.parent.parent


def load_and_validate_data(data_path):
    """
    Load data with proper error handling and validation.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Clean whitespace from all string columns
        string_columns = data.select_dtypes(include=['object']).columns
        for col in string_columns:
            data[col] = data[col].astype(str).str.strip()
        print(f"✓ Cleaned whitespace from {len(string_columns)} categorical columns")
        
        # Handle missing values represented as '?'
        data = data.replace('?', np.nan)
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠ Warning: Found {missing_count} missing values (including '?' markers)")
            missing_by_col = data.isnull().sum()
            for col, count in missing_by_col[missing_by_col > 0].items():
                print(f"  {col}: {count} missing values")
        
        # Validate required columns
        if 'salary' not in data.columns:
            raise ValueError("Target column 'salary' not found in dataset")
        
        return data
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)


def explore_data(data):
    """
    Perform basic exploratory data analysis.
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {data.shape}")
    print(f"\nTarget variable distribution:")
    print(data['salary'].value_counts())
    print(f"\nTarget variable proportions:")
    print(data['salary'].value_counts(normalize=True))
    
    print(f"\nData types:")
    print(data.dtypes.value_counts())
    
    print(f"\nNumerical columns summary:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(data[numerical_cols].describe())
    
    print(f"\nCategorical columns:")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != 'salary':  # Skip target variable
            unique_count = data[col].nunique()
            print(f"  {col}: {unique_count} unique values")


def create_preprocessing_pipeline(X):
    """
    Create a comprehensive preprocessing pipeline.
    """
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nPreprocessing pipeline:")
    print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # Create transformers
    transformers = []
    
    if categorical_cols:
        # Pipeline for categorical features: impute missing values, then one-hot encode
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_cols))
    
    if numerical_cols:
        # Pipeline for numerical features: impute missing values, then scale
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, numerical_cols))
    
    if not transformers:
        raise ValueError("No valid columns found for preprocessing")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any remaining columns
    )
    
    return preprocessor


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Train model with comprehensive evaluation.
    """
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline using only training data to avoid data leakage
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # Create full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    
    # Train model
    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model_pipeline, {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores
    }


def save_model(model, model_path):
    """
    Save the trained model.
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")


def main():
    """
    Main execution function.
    """
    print("Adult Income Prediction - ML Pipeline")
    print("="*50)
    
    try:
        # Get project root
        project_root = get_project_root()
        print(f"Project root: {project_root}")
        
        # Load and validate data
        raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
        data = load_and_validate_data(raw_data_file)
        
        # Explore data
        explore_data(data)
        
        # Prepare features and target
        X = data.drop('salary', axis=1)
        y = data['salary']
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(X, y)
        
        # Save model
        model_path = os.path.join(project_root, "No1", "trained_model.pkl")
        save_model(model, model_path)
        
        print(f"\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()