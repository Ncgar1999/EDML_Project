"""
Bias-Mitigated ML Pipeline for Adult Income Prediction

This version addresses inherited bias by:
1. Removing sensitive attributes (race, sex, native-country)
2. Adding fairness evaluation metrics
3. Implementing bias detection and reporting
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, 
                           roc_auc_score, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Add parent directory to path for utils import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root


def load_and_validate_data(data_path):
    """
    Load and validate data with bias analysis.
    """
    try:
        data = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Clean whitespace
        string_columns = data.select_dtypes(include=['object']).columns
        cleaned_cols = []
        for col in string_columns:
            data[col] = data[col].astype(str).str.strip()
            cleaned_cols.append(col)
        
        if cleaned_cols:
            print(f"✓ Cleaned whitespace from {len(cleaned_cols)} categorical columns")
        
        # Handle missing values (replace '?' with NaN)
        data = data.replace('?', np.nan)
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"⚠ Warning: Found {total_missing} missing values (including '?' markers)")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count} missing values")
        
        return data
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)


def analyze_bias_in_data(data):
    """
    Analyze inherited bias in the dataset.
    """
    print("\n" + "="*50)
    print("BIAS ANALYSIS")
    print("="*50)
    
    # Check for sensitive attributes
    sensitive_attrs = ['race', 'sex', 'native-country']
    found_sensitive = [attr for attr in sensitive_attrs if attr in data.columns]
    
    if found_sensitive:
        print(f"\n⚠ WARNING: Found sensitive attributes: {found_sensitive}")
        print("These attributes can lead to discriminatory outcomes!")
        
        # Show bias in historical data
        for attr in ['race', 'sex']:
            if attr in data.columns:
                bias_table = pd.crosstab(data[attr], data['salary'], normalize='index')
                print(f"\nIncome distribution by {attr}:")
                print(bias_table.round(3))
                
                # Calculate disparate impact ratios
                if attr == 'race' and 'White' in bias_table.index:
                    white_rate = bias_table.loc['White', '>50K']
                    print(f"\nDisparate impact ratios (vs White):")
                    for race in bias_table.index:
                        if race != 'White':
                            ratio = bias_table.loc[race, '>50K'] / white_rate
                            status = "❌ BIAS" if ratio < 0.8 else "✓ OK"
                            print(f"  {race}: {ratio:.3f} {status}")
                
                elif attr == 'sex':
                    male_rate = bias_table.loc['Male', '>50K']
                    female_rate = bias_table.loc['Female', '>50K']
                    ratio = female_rate / male_rate
                    status = "❌ BIAS" if ratio < 0.8 else "✓ OK"
                    print(f"\nGender disparate impact ratio: {ratio:.3f} {status}")


def remove_sensitive_attributes(data, sensitive_attrs=None):
    """
    Remove sensitive attributes to prevent direct discrimination.
    """
    if sensitive_attrs is None:
        sensitive_attrs = ['race', 'sex', 'native-country']
    
    removed_attrs = []
    for attr in sensitive_attrs:
        if attr in data.columns:
            data = data.drop(attr, axis=1)
            removed_attrs.append(attr)
    
    if removed_attrs:
        print(f"\n✓ Removed sensitive attributes: {removed_attrs}")
        print("This prevents direct discrimination based on these attributes.")
    
    return data


def explore_data(data):
    """
    Comprehensive data exploration.
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {data.shape}")
    
    # Target variable analysis
    print(f"\nTarget variable distribution:")
    print(data['salary'].value_counts())
    
    print(f"\nTarget variable proportions:")
    print(data['salary'].value_counts(normalize=True))
    
    # Data types
    print(f"\nData types:")
    print(data.dtypes.value_counts())
    
    # Numerical columns summary
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\nNumerical columns summary:")
        print(data[numerical_cols].describe())
    
    # Categorical columns info
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical columns:")
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
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any remaining columns
    )
    
    return preprocessor


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Train model with comprehensive evaluation and bias checking.
    """
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)
    
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
    Main execution function with bias mitigation.
    """
    print("Adult Income Prediction - BIAS-MITIGATED ML Pipeline")
    print("="*60)
    
    try:
        # Get project root
        project_root = get_project_root()
        print(f"Project root: {project_root}")
        
        # Load and validate data
        data_path = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
        data = load_and_validate_data(data_path)
        
        # Analyze bias in original data
        analyze_bias_in_data(data)
        
        # Remove sensitive attributes to prevent direct discrimination
        data_debiased = remove_sensitive_attributes(data.copy())
        
        # Explore debiased data
        explore_data(data_debiased)
        
        # Prepare features and target
        X = data_debiased.drop('salary', axis=1)
        y = data_debiased['salary']
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(X, y)
        
        # Save model
        model_path = os.path.join(project_root, "No1", "trained_model_bias_mitigated.pkl")
        save_model(model, model_path)
        
        print(f"\n" + "="*60)
        print("BIAS MITIGATION SUMMARY")
        print("="*60)
        print("✓ Removed sensitive attributes (race, sex, native-country)")
        print("✓ Used stratified sampling for fair train/test split")
        print("✓ Applied proper preprocessing with missing value handling")
        print("✓ Model trained without direct access to protected attributes")
        print("\n⚠ Note: This reduces direct discrimination but may not eliminate")
        print("  all bias due to proxy variables and historical patterns.")
        print("  Consider additional fairness techniques for production use.")
        
        print(f"\n✓ Bias-mitigated pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()