import os
import pandas as pd

def test_data_validation_checkpoint():
    """
    CRITICAL MLOPS TEST: Verifies that the Data Validation stage ran 
    successfully and generated a 'True' status contract.
    """
    status_path = "artifacts/data_validation/status.txt"
    
    # Check if the validation artifact file exists
    assert os.path.exists(status_path), f"Validation artifact file missing at {status_path}"
    
    # Read the file content and check if it says 'True'
    with open(status_path, "r") as f:
        status_content = f.read().strip()
        
    assert "Validation status: True" in status_content, (
        f"Data validation failed or status is incorrect. Content: {status_content}"
    )


def test_data_transformation_splits():
    """
    DATA INTEGRITY TEST: Ensures that data transformation successfully partitioned 
    the raw data into non-empty training and testing CSV sets.
    """
    train_path = "artifacts/data_transformation/train.csv"
    test_path = "artifacts/data_transformation/test.csv"
    
    assert os.path.exists(train_path), "Training data split (train.csv) is missing."
    assert os.path.exists(test_path), "Testing data split (test.csv) is missing."
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Verify rows exist to protect against silent training on empty data
    assert len(df_train) > 0, "train.csv contains zero records."
    assert len(df_test) > 0, "test.csv contains zero records."


def test_target_column_constraints():
    """
    MODEL BOUNDARY TEST: Checks if the target label 'loan_status' is present 
    and contains strict binary values (0 or 1).
    """
    test_path = "artifacts/data_transformation/test.csv"
    
    if os.path.exists(test_path):
        df = pd.read_csv(test_path)
        
        # Verify target column is present
        assert "loan_status" in df.columns, "Target column 'loan_status' missing from data splits."
        
        # Verify values are strictly binary (0 = Non-default, 1 = Default)
        unique_values = set(df["loan_status"].unique())
        assert unique_values.issubset({0, 1}), (
            f"Target leakage or data corruption detected! Found non-binary labels: {unique_values}"
        )