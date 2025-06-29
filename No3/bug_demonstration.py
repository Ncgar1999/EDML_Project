"""
Demonstration of the LabelEncoder Bug in Original ML-Pipeline-3.py

This script demonstrates why reusing a single LabelEncoder instance
for multiple categorical columns is problematic.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def demonstrate_labelencoder_bug():
    """
    Shows the problem with reusing LabelEncoder across multiple columns
    """
    print("=" * 60)
    print("DEMONSTRATING LABELENCODER BUG")
    print("=" * 60)
    
    # Create sample data similar to adult dataset
    data = pd.DataFrame({
        'workclass': ['Private', 'Self-emp', 'Private', 'Government'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Bachelors'],
        'marital_status': ['Married', 'Single', 'Married', 'Divorced'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    })
    
    print("Original Data:")
    print(data)
    print()
    
    # Method 1: BUGGY - Reusing single LabelEncoder (original script approach)
    print("METHOD 1: BUGGY - Reusing Single LabelEncoder")
    print("-" * 50)
    
    data_buggy = data.copy()
    le_buggy = LabelEncoder()
    
    for column in data_buggy.columns:
        if data_buggy[column].dtype == 'object':
            print(f"Encoding {column}...")
            data_buggy[column] = le_buggy.fit_transform(data_buggy[column])
            print(f"  Encoder classes after {column}: {le_buggy.classes_}")
    
    print("\nBuggy Result:")
    print(data_buggy)
    print(f"Final encoder classes: {le_buggy.classes_}")
    print("❌ Problem: Encoder only remembers the LAST column's classes!")
    print()
    
    # Method 2: CORRECT - Separate LabelEncoder for each column
    print("METHOD 2: CORRECT - Separate LabelEncoder for Each Column")
    print("-" * 50)
    
    data_correct = data.copy()
    encoders = {}
    
    for column in data_correct.columns:
        if data_correct[column].dtype == 'object':
            encoder = LabelEncoder()  # New encoder for each column
            data_correct[column] = encoder.fit_transform(data_correct[column])
            encoders[column] = encoder
            print(f"Encoded {column}: {encoder.classes_}")
    
    print("\nCorrect Result:")
    print(data_correct)
    print()
    
    # Demonstrate the problem when trying to decode
    print("DECODING DEMONSTRATION")
    print("-" * 30)
    
    try:
        print("Trying to decode with buggy encoder...")
        # This will fail or give wrong results
        decoded_workclass = le_buggy.inverse_transform([0, 1])
        print(f"Decoded workclass: {decoded_workclass}")
        print("⚠️  This is wrong! It's using salary classes for workclass!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nDecoding with correct encoders...")
    decoded_workclass = encoders['workclass'].inverse_transform([0, 1, 2])
    decoded_education = encoders['education'].inverse_transform([0, 1, 2])
    print(f"✅ Workclass classes: {decoded_workclass}")
    print(f"✅ Education classes: {decoded_education}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("❌ Buggy approach: Cannot properly decode or apply to new data")
    print("✅ Correct approach: Each column has its own encoder with proper mappings")
    print("🔧 Fix: Use separate LabelEncoder instance for each categorical column")

def demonstrate_data_leakage():
    """
    Shows the data leakage problem in the original script
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING DATA LEAKAGE ISSUE")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    print("Sample data shape:", data.shape)
    print("Category distribution:", data['category'].value_counts().to_dict())
    print()
    
    # Method 1: BUGGY - Encode before split (data leakage)
    print("METHOD 1: BUGGY - Encoding Before Train/Test Split")
    print("-" * 50)
    
    from sklearn.model_selection import train_test_split
    
    data_leaky = data.copy()
    le_leaky = LabelEncoder()
    data_leaky['category'] = le_leaky.fit_transform(data_leaky['category'])
    
    X_leaky = data_leaky[['feature1', 'category']]
    y_leaky = data_leaky['target']
    X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = train_test_split(
        X_leaky, y_leaky, test_size=0.2, random_state=42
    )
    
    print(f"Encoder saw ALL data: {le_leaky.classes_}")
    print("❌ Problem: Test data was used to fit the encoder!")
    print()
    
    # Method 2: CORRECT - Encode after split
    print("METHOD 2: CORRECT - Encoding After Train/Test Split")
    print("-" * 50)
    
    X = data[['feature1', 'category']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit encoder only on training data
    le_correct = LabelEncoder()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    X_train_encoded['category'] = le_correct.fit_transform(X_train['category'])
    
    # Handle potential unseen categories in test data
    test_categories = X_test['category']
    encoded_test = []
    for cat in test_categories:
        if cat in le_correct.classes_:
            encoded_test.append(le_correct.transform([cat])[0])
        else:
            encoded_test.append(-1)  # Unknown category
    
    X_test_encoded['category'] = encoded_test
    
    print(f"Encoder only saw training data: {le_correct.classes_}")
    print("✅ Correct: Test data was not used to fit the encoder!")
    print()
    
    print("SUMMARY:")
    print("❌ Data leakage: Encoder fitted on entire dataset")
    print("✅ No leakage: Encoder fitted only on training data")
    print("🔧 Fix: Always fit preprocessing on training data only")

if __name__ == "__main__":
    demonstrate_labelencoder_bug()
    demonstrate_data_leakage()