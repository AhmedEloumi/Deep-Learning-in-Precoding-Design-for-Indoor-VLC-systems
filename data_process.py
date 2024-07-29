from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 

def split_data(df, feature_column, label_column, test_size=0.2, random_state=42):
    """
    Preprocesses the data by extracting features and labels, splitting into training and testing sets, and standardizing the data.
    
    Parameters:
    - df: DataFrame containing the data
    - feature_column: Name of the column containing features
    - label_column: Name of the column containing labels
    - test_size: Size of the test set (default is 0.2)
    - random_state: Random seed for reproducibility (default is 42)
    
    Returns:
    - X_train_reshaped: Reshaped and standardized training features
    - X_test_reshaped: Reshaped and standardized testing features
    - y_train_scaled: Scaled training labels
    - y_test_scaled: Scaled testing labels
    """
    # Extract features (X) and labels (y)
    X = np.stack(df[feature_column].values)
    y = np.stack(df[label_column].values)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Reshape the features for compatibility with Conv1D layers
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return  X_train, X_train_scaled, X_train_reshaped, X_test, X_test_scaled, X_test_reshaped, y_train, y_train_scaled, y_test, y_test_scaled


def divide_data(df, feature_column, label_column, test_size=0.1, valid_size=0.2, random_state=42, shuffle=True):
    
    # Extract features (X) and labels (y)
    X = np.stack(df[feature_column].values)
    y = np.stack(df[label_column].values)
    
    # Split the data into train+valid and test sets
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    # Calculate the adjusted validation size for the train+valid split
    adjusted_valid_size = valid_size / (1 - test_size)
    
    # Split the train+valid set into train and valid sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=adjusted_valid_size, random_state=random_state, shuffle=shuffle)
    
    # Standardize the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Reshape the features for compatibility with Conv1D layers
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_valid_reshaped = X_valid_scaled.reshape(X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return (X_train, X_train_scaled, X_train_reshaped, 
            X_valid, X_valid_scaled, X_valid_reshaped, 
            X_test, X_test_scaled, X_test_reshaped, 
            y_train, y_train_scaled, y_valid, y_valid_scaled, y_test, y_test_scaled)


