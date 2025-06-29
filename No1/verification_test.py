"""
Verification script to test both original and improved ML pipelines
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a script and capture its output"""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("="*60)
    print("VERIFICATION TEST: ML Pipeline Scripts")
    print("="*60)
    
    # Test original script
    print("\n1. Testing Original Script (ML-Pipeline-1.py):")
    print("-" * 50)
    success, stdout, stderr = run_script("ML-Pipeline-1.py")
    
    if success:
        print("✅ Original script runs successfully")
        # Extract accuracy from output
        lines = stdout.split('\n')
        for line in lines:
            if 'accuracy' in line and '0.' in line:
                print(f"   Accuracy: {line.strip()}")
                break
    else:
        print("❌ Original script failed")
        print(f"Error: {stderr}")
    
    # Test improved script
    print("\n2. Testing Improved Script (ML-Pipeline-1-improved.py):")
    print("-" * 50)
    success, stdout, stderr = run_script("ML-Pipeline-1-improved.py")
    
    if success:
        print("✅ Improved script runs successfully")
        # Extract key metrics
        lines = stdout.split('\n')
        for line in lines:
            if 'Test Accuracy:' in line:
                print(f"   {line.strip()}")
            elif 'ROC AUC Score:' in line:
                print(f"   {line.strip()}")
            elif 'Cross-validation accuracy:' in line:
                print(f"   {line.strip()}")
    else:
        print("❌ Improved script failed")
        print(f"Error: {stderr}")
    
    # Check if model was saved
    print("\n3. Checking Model Persistence:")
    print("-" * 50)
    if os.path.exists("trained_model.pkl"):
        print("✅ Model file saved successfully")
        print(f"   File size: {os.path.getsize('trained_model.pkl')} bytes")
    else:
        print("❌ Model file not found")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()