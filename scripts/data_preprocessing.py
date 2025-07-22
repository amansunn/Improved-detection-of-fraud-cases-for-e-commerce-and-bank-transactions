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
print(fraud_data.describe())
print(fraud_data['is_fraud'].value_counts())

# EDA: Bivariate analysis (example)
print(fraud_data.groupby('is_fraud').mean())

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
X = fraud_data.drop(['is_fraud'], axis=1)
y = fraud_data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Analyze class distribution
print('Class distribution in training set:')
print(y_train.value_counts())

# Apply SMOTE for oversampling (justification: preserves all minority class samples, generates synthetic examples)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train.select_dtypes(include=[np.number]), y_train)

# Normalization and Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

# Encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# Save processed data (optional)
pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test_scaled.csv', index=False)
pd.DataFrame(X_train_cat).to_csv('data/processed/X_train_cat.csv', index=False)
pd.DataFrame(X_test_cat).to_csv('data/processed/X_test_cat.csv', index=False)
pd.DataFrame(y_train_res).to_csv('data/processed/y_train_res.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

print('Data preprocessing complete.')
