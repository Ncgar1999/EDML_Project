"""
Improved ML Pipeline for Adult Dataset Income Prediction

This script addresses the bugs and issues found in the original ML-Pipeline-3.py:
1. Fixes LabelEncoder reuse bug
2. Prevents data leakage
3. Adds proper data validation
4. Improves feature engineering
5. Enhances model evaluation
6. Adds error handling and documentation
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root


class AdultDatasetPipeline:
    """
    ML Pipeline for Adult Dataset Income Prediction
    
    Fixes the bugs in the original script and adds comprehensive
    data processing, model training, and evaluation capabilities.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the pipeline
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the dataset
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
                
            data = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully: {data.shape}")
            
            # Basic validation
            if data.empty:
                raise ValueError("Dataset is empty")
                
            if 'salary' not in data.columns:
                raise ValueError("Target column 'salary' not found")
                
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                print(f"⚠ Missing values found:\n{missing_values[missing_values > 0]}")
            else:
                print("✓ No missing values found")
                
            # Check for duplicates
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                print(f"⚠ Found {duplicates} duplicate rows")
                data = data.drop_duplicates()
                print(f"✓ Duplicates removed. New shape: {data.shape}")
            else:
                print("✓ No duplicates found")
                
            return data
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
            
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Improved feature preprocessing
        
        Args:
            data: Raw dataset
            
        Returns:
            Preprocessed dataset
        """
        data = data.copy()
        
        # Instead of dropping education, let's keep it and drop education-num
        # since education is more interpretable
        if 'education-num' in data.columns:
            data = data.drop(columns=['education-num'])
            print("✓ Dropped redundant 'education-num' column")
            
        # Keep occupation as it's valuable for income prediction
        # Only drop if there are too many categories or missing values
        if 'occupation' in data.columns:
            occupation_categories = data['occupation'].nunique()
            print(f"✓ Keeping 'occupation' column with {occupation_categories} categories")
            
        return data
        
    def encode_categorical_features(self, X_train: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Properly encode categorical features without data leakage
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Encoded training and test features
        """
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            # Create a new encoder for each column
            encoder = LabelEncoder()
            
            # Fit only on training data
            X_train_encoded[column] = encoder.fit_transform(X_train[column])
            
            # Handle unseen categories in test data
            def safe_transform(values):
                result = []
                for value in values:
                    if value in encoder.classes_:
                        result.append(encoder.transform([value])[0])
                    else:
                        # Assign a default value for unseen categories
                        result.append(-1)  # or use most frequent class
                return np.array(result)
            
            X_test_encoded[column] = safe_transform(X_test[column])
            
            # Store encoder for future use
            self.label_encoders[column] = encoder
            
            print(f"✓ Encoded '{column}': {len(encoder.classes_)} categories")
            
        return X_train_encoded, X_test_encoded
        
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare data for training with proper preprocessing
        
        Args:
            data: Preprocessed dataset
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = data.drop(columns=['salary'])
        y = data['salary']
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.label_encoders['salary'] = target_encoder
        
        print(f"✓ Target classes: {target_encoder.classes_}")
        print(f"✓ Class distribution: {np.bincount(y_encoded)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=self.random_state, stratify=y_encoded
        )
        
        print(f"✓ Data split: Train {X_train.shape}, Test {X_test.shape}")
        
        # Encode categorical features properly
        X_train_encoded, X_test_encoded = self.encode_categorical_features(X_train, X_test)
        
        # Store feature names
        self.feature_names = X_train_encoded.columns.tolist()
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train_encoded)
        X_test_scaled = self.scaler.transform(X_test_encoded)
        
        print("✓ Features scaled")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train the Random Forest model with better parameters
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        # Use better hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("🔄 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        return self.model
        
    def evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        results['roc_auc'] = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"ROC AUC Score: {results['roc_auc']:.4f}")
        print(f"Cross-validation: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        
        # Classification report
        print("\nClassification Report:")
        target_names = self.label_encoders['salary'].classes_
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return results
        
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze and display feature importance
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        print(importance_df.head(10))
        
        return importance_df
        
    def run_pipeline(self, data_file: str) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            data_file: Path to the data file
            
        Returns:
            Dictionary with all results
        """
        print("🚀 Starting ML Pipeline...")
        print("="*50)
        
        try:
            # Load and validate data
            data = self.load_and_validate_data(data_file)
            
            # Preprocess features
            data = self.preprocess_features(data)
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(data)
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            results = self.evaluate_model(X_train, X_test, y_train, y_test)
            
            # Analyze feature importance
            importance_df = self.analyze_feature_importance()
            results['feature_importance'] = importance_df
            
            print("\n✅ Pipeline completed successfully!")
            return results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the improved pipeline"""
    # Get data file path
    project_root = get_project_root()
    data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
    
    # Run pipeline
    pipeline = AdultDatasetPipeline(random_state=42)
    results = pipeline.run_pipeline(data_file)
    
    return results


if __name__ == "__main__":
    main()