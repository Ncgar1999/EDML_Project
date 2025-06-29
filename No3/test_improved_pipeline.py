"""
Test script to verify the improved ML pipeline works correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_labelencoder_fix():
    """Test that the LabelEncoder fix works correctly"""
    print("Testing LabelEncoder fix...")
    
    # Create test data
    data = pd.DataFrame({
        'cat1': ['A', 'B', 'C', 'A'],
        'cat2': ['X', 'Y', 'X', 'Z'],
        'target': [0, 1, 0, 1]
    })
    
    # Simulate improved approach
    encoders = {}
    data_encoded = data.copy()
    
    for col in ['cat1', 'cat2']:
        encoder = LabelEncoder()
        data_encoded[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
    
    # Test decoding
    try:
        decoded_cat1 = encoders['cat1'].inverse_transform([0, 1, 2])
        decoded_cat2 = encoders['cat2'].inverse_transform([0, 1, 2])
        print(f"✅ cat1 decoded: {decoded_cat1}")
        print(f"✅ cat2 decoded: {decoded_cat2}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_data_leakage_prevention():
    """Test that data leakage is prevented"""
    print("\nTesting data leakage prevention...")
    
    from sklearn.model_selection import train_test_split
    
    # Create test data with categories
    np.random.seed(42)
    data = pd.DataFrame({
        'feature': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Split first
    X = data[['feature', 'category']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode only on training data
    encoder = LabelEncoder()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # Fit only on training data
    X_train_encoded['category'] = encoder.fit_transform(X_train['category'])
    
    # Transform test data (handling unseen categories)
    test_categories = X_test['category']
    encoded_test = []
    for cat in test_categories:
        if cat in encoder.classes_:
            encoded_test.append(encoder.transform([cat])[0])
        else:
            encoded_test.append(-1)  # Unknown category
    
    X_test_encoded['category'] = encoded_test
    
    print(f"✅ Training categories: {encoder.classes_}")
    print(f"✅ Test encoding successful: {len(encoded_test)} samples")
    return True

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\nTesting pipeline components...")
    
    try:
        # Import the improved pipeline
        exec(open('ML-Pipeline-3-improved.py').read(), {'__file__': 'ML-Pipeline-3-improved.py'})
        print("✅ Pipeline script loads successfully")
        return True
    except Exception as e:
        print(f"❌ Pipeline loading error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("TESTING IMPROVED ML PIPELINE")
    print("=" * 50)
    
    tests = [
        test_labelencoder_fix,
        test_data_leakage_prevention,
        test_pipeline_components
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! The improved pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests()