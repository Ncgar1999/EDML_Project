"""
Fixed version of ML-Pipeline-1.py that preserves original data processing behavior
but fixes only the actual bugs (missing utils module and convergence warning).
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path


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


# Get project root (fixes missing utils module)
project_root = get_project_root()

# Load data (preserves original behavior - no whitespace cleaning)
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Prepare features and target (preserves original behavior)
X = data.drop('salary', axis=1)
y = data['salary']

# Get categorical columns (preserves original behavior)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessor with scaling for numerical features (fixes convergence warning)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)  # Added to fix convergence
    ],
    remainder='passthrough'
)

# Create pipeline (preserves original structure)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))  # Preserves original random behavior
])

# Train/test split (preserves original behavior)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (preserves original behavior)
model_pipeline.fit(X_train, y_train)

# Evaluate model (preserves original behavior)
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))