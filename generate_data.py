import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
n_samples = 2500

# 1. Generate realistic features
# Age between 20 and 70
person_age = np.random.randint(20, 70, n_samples).astype(float)

# Income using a log-normal distribution to simulate real-world wealth distribution (mostly 40k-100k, some outliers)
person_income = np.random.lognormal(mean=11.2, sigma=0.5, size=n_samples).round(2)

# Loan amounts between $1,000 and $40,000
loan_amnt = np.random.randint(1000, 40000, n_samples).astype(float)

# Interest rates between 5% and 23%
loan_int_rate = np.random.uniform(5.0, 23.0, n_samples).round(2)

# Credit history length (highly correlated with age: you can't have a 30-year history if you are 25)
cb_person_cred_hist_length = np.maximum(
    1, (person_age - 18) * np.random.uniform(0.2, 0.9, n_samples)
).round().astype(float)


# 2. Simulate Realistic Loan Defaults (Target Variable)
# Create a hidden "risk score" based on financial logic
# High loan-to-income ratio + high interest rate = High Risk
loan_to_income_ratio = loan_amnt / person_income
risk_score = (loan_to_income_ratio * 15) + (loan_int_rate / 15)

# Convert risk score to a probability using a sigmoid-like function
prob_default = 1 / (1 + np.exp(-(risk_score - np.mean(risk_score))))

# Add some randomness (noise) and assign 1 (Default) or 0 (Safe)
# We multiply by 0.6 to create an imbalanced dataset (~20-25% default rate), which is standard for credit risk.
loan_status = (np.random.rand(n_samples) < (prob_default * 0.6)).astype(int)

# 3. Create DataFrame and export
df = pd.DataFrame({
    'person_age': person_age,
    'person_income': person_income,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'loan_status': loan_status
})

# Save to CSV
file_name = 'credit_risk.csv'
df.to_csv(file_name, index=False)

print(f"✅ Successfully generated {file_name} with {len(df)} rows.")
print(f"Target distribution (0 = Safe, 1 = Default):\n{df['loan_status'].value_counts(normalize=True) * 100}")