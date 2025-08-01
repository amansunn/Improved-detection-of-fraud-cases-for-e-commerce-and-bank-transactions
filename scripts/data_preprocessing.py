"""
Script: data_preprocessing.py
Purpose: Data analysis and preprocessing for fraud detection project.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load data
fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
ip_country = pd.read_csv('data/raw/IpAddress_to_Country.csv')

# Handle missing values
fraud_data = fraud_data.dropna(axis=0, how='any')  # Drop rows with missing values

# Data cleaning
fraud_data = fraud_data.drop_duplicates()

# Correct data types (example: convert date columns)
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])


# EDA: Univariate analysis (example)
print(f"Columns in fraud_data: {fraud_data.columns.tolist()}")
print(fraud_data.describe())
if 'class' in fraud_data.columns:
    print(fraud_data['class'].value_counts())
    # EDA: Bivariate analysis (example)
    numeric_cols = fraud_data.select_dtypes(include=[np.number]).columns
    print(fraud_data.groupby('class')[numeric_cols].mean())
else:
    print("Column 'class' not found. Please check your CSV file for the correct column name.")

# Convert IP addresses to integer format for merging
import ipaddress
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return np.nan
fraud_data['ip_int'] = fraud_data['ip_address'].apply(ip_to_int)
ip_country['lower_bound_ip_int'] = ip_country['lower_bound_ip_address'].apply(ip_to_int)
ip_country['upper_bound_ip_int'] = ip_country['upper_bound_ip_address'].apply(ip_to_int)

# Merge Fraud_Data with IpAddress_to_Country
# For each transaction, find the country where ip_int is between lower and upper bound
ip_country = ip_country[['country', 'lower_bound_ip_int', 'upper_bound_ip_int']]
def find_country(ip):
    row = ip_country[(ip_country['lower_bound_ip_int'] <= ip) & (ip_country['upper_bound_ip_int'] >= ip)]
    if not row.empty:
        return row.iloc[0]['country']
    return np.nan
fraud_data['country'] = fraud_data['ip_int'].apply(find_country)

# Feature Engineering
# Transaction frequency and velocity
fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
fraud_data['transaction_velocity'] = fraud_data.groupby('user_id')['purchase_time'].transform(lambda x: x.diff().dt.total_seconds().fillna(0))

# Time-based features
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()


# Data Transformation
# Handle class imbalance
X = fraud_data.drop(['class'], axis=1)
y = fraud_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Analyze class distribution
print('Class distribution in training set:')
print(y_train.value_counts())




# Impute missing values in numeric columns before SMOTE
# Exclude 'ip_int' and 'country' from numeric features for SMOTE/scaling
exclude_cols = ['ip_int', 'country']
X_train_num = X_train.select_dtypes(include=[np.number]).copy()
X_train_num = X_train_num.drop(columns=[col for col in exclude_cols if col in X_train_num.columns], errors='ignore')
if X_train_num.isnull().any().any():
    print('Imputing missing values in numeric features with column mean...')
    X_train_num = X_train_num.fillna(X_train_num.mean())
    # If any columns are still NaN (e.g., all values were NaN), fill with 0
    if X_train_num.isnull().any().any():
        print('Columns still containing NaN after mean imputation:', X_train_num.columns[X_train_num.isnull().any()].tolist())
        X_train_num = X_train_num.fillna(0)

# Apply SMOTE for oversampling (justification: preserves all minority class samples, generates synthetic examples)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_num, y_train)


# Normalization and Scaling
scaler = StandardScaler()
X_test_num = X_test.select_dtypes(include=[np.number]).copy()
X_test_num = X_test_num.drop(columns=[col for col in exclude_cols if col in X_test_num.columns], errors='ignore')
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_num)



# Encode categorical features, excluding high-cardinality columns
high_cardinality_cols = ['device_id', 'ip_address', 'user_id']
categorical_cols = [col for col in X_train.select_dtypes(include=['object', 'category']).columns if col not in high_cardinality_cols]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
if categorical_cols:
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])
else:
    X_train_cat = np.empty((X_train.shape[0], 0))
    X_test_cat = np.empty((X_test.shape[0], 0))

# Ensure processed directory exists
import os
os.makedirs('data/processed', exist_ok=True)

import os
processed_dir = os.path.abspath('data/processed')
print(f'Saving processed data to {processed_dir} ...')
try:
    pd.DataFrame(X_train_scaled).to_csv(os.path.join(processed_dir, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_scaled).to_csv(os.path.join(processed_dir, 'X_test_scaled.csv'), index=False)
    pd.DataFrame(X_train_cat).to_csv(os.path.join(processed_dir, 'X_train_cat.csv'), index=False)
    pd.DataFrame(X_test_cat).to_csv(os.path.join(processed_dir, 'X_test_cat.csv'), index=False)
    pd.DataFrame(y_train_res).to_csv(os.path.join(processed_dir, 'y_train_res.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    print('Processed data saved successfully.')
    print('Files in processed directory:')
    print(os.listdir(processed_dir))
except Exception as e:
    print(f'Error saving processed data: {e}')

print('Data preprocessing complete.')
