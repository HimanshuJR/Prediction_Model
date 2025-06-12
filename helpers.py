import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Stub: Load CSV data
    return pd.read_csv(file_path, low_memory=False)

def preprocess_data(df):
    # Stub: Example preprocessing (scaling numeric, encoding categorical)
    scaler = StandardScaler()
    encoders = {}
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler, encoders

def create_time_features(df, timestamp_column):
    # Stub: Add time-based features
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df['hour'] = df[timestamp_column].dt.hour
    df['dayofweek'] = df[timestamp_column].dt.dayofweek
    df['month'] = df[timestamp_column].dt.month
    return df

def prepare_training_data(df, target_column):
    # Stub: Split into train/test
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42) 