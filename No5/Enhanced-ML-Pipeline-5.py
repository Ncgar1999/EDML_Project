import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def load_and_explore_data(file_path):
    """Load and explore the dataset."""
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Basic exploration
    print(f"Dataset shape: {data.shape}")
    print("\nMissing values per column:")
    print(data.isnull().sum())
    
    print("\nUnique values in key categorical columns:")
    for col in ['sex', 'race', 'c_charge_degree', 'score_text']:
        if col in data.columns:
            print(f"{col}: {data[col].unique()}")
    
    print("\nClass distribution:")
    if 'score_text' in data.columns:
        print(data['score_text'].value_counts(normalize=True))
    
    # Check for potential data leakage
    print("\nChecking for potential data leakage...")
    if 'decile_score' in data.columns and 'score_text' in data.columns:
        print("\nRelationship between decile_score and score_text:")
        print(pd.crosstab(data['decile_score'], data['score_text'], normalize='index'))
        print("\nThis shows that decile_score directly determines score_text, which would cause data leakage.")
    
    return data

def check_feature_importance(X, y, feature_names):
    """Check feature importance to identify potential data leakage."""
    # Train a simple random forest to check feature importance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Define feature types
    numeric_features = ['age', 'priors_count', 'days_b_screening_arrest', 'jail_time_days']
    binary_features = ['is_recid', 'two_year_recid']
    categorical_features = ['sex', 'c_charge_degree', 'race']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('bin', SimpleImputer(strategy='most_frequent'), binary_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    rf = model.named_steps['classifier']
    
    # Get the feature names after preprocessing
    cat_features = []
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            unique_values = X[cat_feature].dropna().unique()
            for value in unique_values:
                cat_features.append(f"{cat_feature}_{value}")
    
    # Combine all feature names
    all_features = numeric_features + binary_features + cat_features
    
    # Get feature importances (may not match exactly with all_features due to one-hot encoding)
    importances = rf.feature_importances_
    
    # Print feature importances
    print("\nFeature Importance (top features):")
    for i in range(min(10, len(importances))):
        print(f"Feature {i+1}: Importance = {importances[i]:.4f}")
    
    print("\nNote: Feature names may not match exactly due to preprocessing transformations.")
    
    return importances

def preprocess_data(data, target_col='score_text', test_size=0.2, random_state=42):
    """Preprocess the data for modeling."""
    # Select relevant columns - removing decile_score which is directly related to score_text
    columns_to_use = ['sex', 'age', 'c_charge_degree', 'race', 'score_text', 
                      'priors_count', 'days_b_screening_arrest', 
                      'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    
    data = data[columns_to_use].copy()
    
    # Process date columns
    for col in ['c_jail_in', 'c_jail_out']:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Create jail time feature
    data['jail_time_days'] = (data['c_jail_out'] - data['c_jail_in']).dt.total_seconds() / (24 * 3600)
    data['jail_time_days'] = data['jail_time_days'].fillna(0)
    
    # Drop original date columns
    data = data.drop(columns=['c_jail_in', 'c_jail_out'])
    
    # Handle target variable
    print("\nOriginal target distribution:")
    print(data[target_col].value_counts())
    
    # Note: Instead of simply replacing 'Medium' with 'Low', we'll keep all classes
    # and use a proper multi-class approach or explain the binary conversion
    
    # For this example, we'll convert to binary as in the original script
    print("\nConverting to binary classification (High vs. non-High)")
    data[target_col] = data[target_col].apply(lambda x: 'High' if x == 'High' else 'Low')
    
    print("\nNew target distribution:")
    print(data[target_col].value_counts())
    
    # Split data
    X = data.drop(columns=[target_col])
    y = label_binarize(data[target_col], classes=['High', 'Low']).ravel()
    
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    """Create a comprehensive preprocessing pipeline."""
    # Define feature groups
    numeric_features = ['age', 'priors_count', 'days_b_screening_arrest', 'jail_time_days']
    binary_features = ['is_recid', 'two_year_recid']
    categorical_features = ['sex', 'c_charge_degree', 'race']
    
    # Create transformers
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Build, train and evaluate the model."""
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Fairness analysis
    fairness_analysis(X_test, y_test, model, y_pred)
    
    return model

def fairness_analysis(X_test, y_test, model, y_pred, sensitive_feature='race'):
    """Analyze model fairness across different demographic groups."""
    if sensitive_feature in X_test.columns:
        groups = X_test[sensitive_feature].unique()
        
        print(f"\nFairness Analysis by {sensitive_feature}:")
        
        # Calculate metrics for each group
        for group in groups:
            mask = X_test[sensitive_feature] == group
            
            if mask.sum() > 0:
                group_y_test = y_test[mask]
                group_y_pred = y_pred[mask]
                
                # Calculate metrics
                print(f"\n{group} (n={mask.sum()}):")
                print(classification_report(group_y_test, group_y_pred))
    else:
        print(f"Warning: {sensitive_feature} not found in test data")

def tune_hyperparameters(X_train, y_train, cv=5):
    """Tune model hyperparameters using cross-validation."""
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Define models to try
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'class_weight': [None, 'balanced']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
        }
    }
    
    best_models = {}
    
    # Tune each model
    for name, config in models.items():
        print(f"\nTuning {name}...")
        
        # Create pipeline with current model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Prepare parameter grid
        param_grid = {f'classifier__{key}': val for key, val in config['params'].items()}
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit and evaluate
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_models

def main():
    # Get project root and data file path
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
    
    # Load and explore data
    print("Step 1: Loading and exploring data...")
    raw_data = load_and_explore_data(raw_data_file)
    
    # Preprocess data
    print("\nStep 2: Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(raw_data)
    
    # Check feature importance to identify potential data leakage
    print("\nStep 3: Checking feature importance...")
    feature_names = X_train.columns.tolist()
    check_feature_importance(X_train, y_train, feature_names)
    
    # Option 1: Simple model building and evaluation
    print("\nStep 4: Building and evaluating baseline model...")
    baseline_model = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Option 2: Hyperparameter tuning
    print("\nStep 5: Tuning model hyperparameters...")
    best_models = tune_hyperparameters(X_train, y_train)
    
    # Evaluate best models
    print("\nStep 6: Evaluating best models...")
    for name, model in best_models.items():
        print(f"\nEvaluating best {name} model:")
        y_pred = model.predict(X_test)
        print(f"Accuracy: {model.score(X_test, y_test):.4f}")
        print(classification_report(y_test, y_pred))
        
        # Fairness analysis for best model
        fairness_analysis(X_test, y_test, model, y_pred)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()