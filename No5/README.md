# Enhanced ML Pipeline for COMPAS Dataset

This directory contains an enhanced version of the original ML pipeline for the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset.

## Improvements Over Original Pipeline

### 1. Data Leakage Detection and Prevention

The original pipeline had a critical data leakage issue:
- It included `decile_score` as a feature, which directly determines the target variable `score_text`
- This resulted in artificially perfect accuracy (100%)

Our enhanced pipeline:
- Identifies and removes the leaking feature (`decile_score`)
- Provides data exploration to detect such issues
- Shows more realistic model performance (around 83-84% accuracy)

### 2. Comprehensive Data Preprocessing

The original pipeline only processed 2 out of 13 loaded columns:
- Only applied preprocessing to 'is_recid' and 'age'
- Ignored important features like 'race', 'sex', 'priors_count', etc.

Our enhanced pipeline:
- Properly processes all relevant features
- Handles categorical variables with appropriate encoding
- Applies proper scaling to numerical features
- Creates meaningful features from date columns

### 3. Improved Model Evaluation

The original pipeline used basic evaluation metrics:
- Only reported accuracy and a basic classification report
- No cross-validation or hyperparameter tuning

Our enhanced pipeline:
- Implements cross-validation for more robust evaluation
- Performs hyperparameter tuning for multiple models
- Includes fairness analysis across demographic groups
- Reports more comprehensive metrics (precision, recall, F1, ROC AUC)

### 4. Better Code Structure and Documentation

The original pipeline was written as a script with no functions:
- Hard to maintain and extend
- Limited error handling
- No documentation

Our enhanced pipeline:
- Organized into modular functions with clear responsibilities
- Includes comprehensive documentation
- Implements proper error handling
- Follows best practices for machine learning pipelines

### 5. Ethical Considerations

The original pipeline didn't address ethical concerns:
- No analysis of potential bias across demographic groups
- No consideration of fairness metrics

Our enhanced pipeline:
- Includes fairness analysis across different racial groups
- Highlights potential disparities in model performance
- Provides a foundation for more in-depth fairness analysis

## Results

The enhanced pipeline achieves:
- More realistic accuracy (~83-84% vs. artificially perfect 100%)
- Better generalization to unseen data
- More comprehensive evaluation of model performance
- Insights into potential bias across demographic groups

## Usage

Run the enhanced pipeline with:

```bash
python Enhanced-ML-Pipeline-5.py
```

## Future Improvements

1. Implement more sophisticated fairness metrics and bias mitigation techniques
2. Add visualization of model performance and feature importance
3. Explore more advanced models beyond Logistic Regression and Random Forest
4. Implement proper handling of class imbalance
5. Add model interpretability analysis