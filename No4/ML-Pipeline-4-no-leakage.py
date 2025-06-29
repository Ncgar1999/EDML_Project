"""
ML Pipeline for Student Grade Prediction (No Data Leakage Version)
This script predicts student final grades (G3) without using intermediate grades (G1, G2).

This version addresses the data leakage issue by excluding G1 and G2 features,
making it suitable for predicting final grades based only on student characteristics
and behaviors, not previous academic performance.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path for utils import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils import get_project_root
except ImportError:
    logger.error("Could not import utils module. Make sure utils.py exists in parent directory.")
    sys.exit(1)


def load_and_validate_data(file_path):
    """Load and validate the dataset."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        logger.info(f"Loading dataset from: {file_path}")
        data = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['G3']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Dataset loaded successfully. Shape: {data.shape}")
        logger.info(f"Target variable (G3) range: {data['G3'].min()} - {data['G3'].max()}")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        else:
            logger.info("No missing values found")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_preprocessing_pipeline(X):
    """Create a preprocessing pipeline for numerical and categorical features."""
    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation for regression model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"\n{model_name} Evaluation Results:")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # Interpretation
    if r2 > 0.8:
        logger.info("Excellent model performance (R² > 0.8)")
    elif r2 > 0.6:
        logger.info("Good model performance (R² > 0.6)")
    elif r2 > 0.4:
        logger.info("Moderate model performance (R² > 0.4)")
    else:
        logger.info("Poor model performance (R² ≤ 0.4)")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def analyze_feature_importance(model, feature_names, top_n=10):
    """Analyze and display feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Most Important Features:")
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows()):
            logger.info(f"{i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        return feature_importance_df
    else:
        logger.warning("Model does not have feature importance information")
        return None


def main():
    """Main execution function."""
    try:
        # Get project root and construct file path
        project_root = get_project_root()
        raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
        
        # Load and validate data
        raw_data = load_and_validate_data(raw_data_file)
        
        # Prepare features and target - EXCLUDE G1 and G2 to avoid data leakage
        exclude_columns = ['G3', 'G1', 'G2']  # Exclude intermediate grades
        X = raw_data.drop(columns=exclude_columns)
        y = raw_data['G3']
        
        logger.info("Data leakage prevention: Excluded G1 and G2 from features")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(X_train)
        
        # Create full pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Hyperparameter tuning (reduced grid for faster execution without G1/G2)
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Cross-validation on training data
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Evaluate model
        train_metrics = evaluate_model(y_train, y_train_pred, "Training Set")
        test_metrics = evaluate_model(y_test, y_test_pred, "Test Set")
        
        # Check for overfitting
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        if r2_diff > 0.1:
            logger.warning(f"Potential overfitting detected (R² difference: {r2_diff:.4f})")
        else:
            logger.info(f"Good generalization (R² difference: {r2_diff:.4f})")
        
        # Feature importance analysis
        preprocessor_fitted = best_model.named_steps['preprocessor']
        
        # Get feature names from the preprocessor
        feature_names = []
        
        # Numerical features
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        feature_names.extend(numerical_cols)
        
        # Categorical features (one-hot encoded)
        cat_features = preprocessor_fitted.named_transformers_['cat']
        if hasattr(cat_features, 'get_feature_names_out'):
            cat_feature_names = cat_features.get_feature_names_out()
            feature_names.extend(cat_feature_names)
        
        # Analyze feature importance
        regressor = best_model.named_steps['regressor']
        feature_importance_df = analyze_feature_importance(regressor, feature_names)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Best Model: Random Forest Regressor (No Data Leakage)")
        logger.info(f"Test R² Score: {test_metrics['r2']:.4f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
        
        # Performance interpretation for no-leakage scenario
        if test_metrics['r2'] > 0.5:
            logger.info("✓ Good performance for predicting grades without previous academic data")
        elif test_metrics['r2'] > 0.3:
            logger.info("✓ Moderate performance - reasonable for this challenging prediction task")
        else:
            logger.info("⚠ Limited predictive power - consider additional feature engineering")
        
        logger.info("\nNote: This model predicts final grades based only on student")
        logger.info("characteristics and behaviors, not previous academic performance.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()