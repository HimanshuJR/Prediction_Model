import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from various file formats
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (preprocessed_data, scaler, encoders)
    """
    # Create copies to avoid modifying original data
    data = df.copy()
    
    # Drop timestamp column if it exists (we'll use created time features instead)
    if 'timestamp' in data.columns:
        data = data.drop('timestamp', axis=1)
    
    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data, scaler, encoders

def prepare_training_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_time_features(df, timestamp_column):
    """
    Create time-based features from timestamp column
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_column (str): Name of the timestamp column
        
    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Extract time features
    df['hour'] = df[timestamp_column].dt.hour
    df['day_of_week'] = df[timestamp_column].dt.dayofweek
    df['day_of_month'] = df[timestamp_column].dt.day
    df['month'] = df[timestamp_column].dt.month
    df['year'] = df[timestamp_column].dt.year
    
    return df 