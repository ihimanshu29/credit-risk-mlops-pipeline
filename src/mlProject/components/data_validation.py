import os
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validateAllColumns(self) -> bool:
        try:
            # Assume data is valid until a STRICT STRUCTURAL anomaly is found
            validation_status = True
            
            # Load incoming data and expected schema
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())
            
            logger.info("Starting comprehensive data validation process...")

            # 1. Structural Check: Look for unexpected (hacker/drift) columns
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logger.error(f"Validation Failed: Unexpected column '{col}' found in dataset.")

            # 2. Structural Check: Look for missing expected columns
            for expected_col in all_schema:
                if expected_col not in all_cols:
                    validation_status = False
                    logger.error(f"Validation Failed: Expected column '{expected_col}' is missing from dataset.")


            # 3. Integrity Check: Identify Missing Values (Nulls/NaNs)
            null_counts = data.isnull().sum().sum()
            if null_counts > 0:
                logger.warning(f"Data Alert: Found {null_counts} missing (null) values. Passing to Transformation Imputer.")

            # 4. Domain Logic Check: Enforce Safe Value Ranges
            if 'person_age' in all_cols:
                # Assuming credit age limits are roughly 18 to 100
                if not data['person_age'].between(18, 100).all():
                    logger.warning("Data Alert: 'person_age' contains values outside 18-100. Passing to Transformation Filter.")

            if 'person_income' in all_cols:
                if (data['person_income'] < 0).any():
                    logger.warning("Data Alert: 'person_income' contains negative values.")

            if 'loan_amnt' in all_cols:
                if (data['loan_amnt'] < 0).any():
                    logger.warning("Data Alert: 'loan_amnt' contains negative values.")


            # 5. Write the final status to the file EXACTLY ONCE
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            # 6. Final logging output
            if validation_status:
                logger.info("Schema validation completed successfully. Structural checks passed.")
            else:
                logger.error("Schema validation failed. Check preceding logs for specific structural errors.")

            return validation_status

        except Exception as e:
            logger.exception("An unhandled exception occurred during the data validation stage.")
            raise e