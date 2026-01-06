"""
Data loading and preprocessing module for diabetes prediction.
"""
import pandas as pd
import numpy as np
import os
import urllib.request

def download_dataset(data_dir='data'):
    """
    Download the Pima Indians Diabetes Dataset if not already present.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'diabetes.csv')
    
    if os.path.exists(filepath):
        print(f"Dataset already exists at {filepath}")
        return filepath
    
    # URL for Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    try:
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset downloaded successfully to {filepath}")
        
        # Add column names
        df = pd.read_csv(filepath, header=None)
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df.to_csv(filepath, index=False)
        print("Column names added to dataset")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the dataset and place it in the data directory")
        return None
    
    return filepath

def load_and_preprocess_data(filepath='data/diabetes.csv'):
    """
    Load and preprocess the diabetes dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        X: Features dataframe
        y: Target series
        df: Full preprocessed dataframe
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nOutcome distribution:\n{df['Outcome'].value_counts()}")
    
    # Handle zero values (0 is invalid for some features)
    # Replace 0 with NaN for features where 0 is impossible
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.nan)
        # Fill NaN with median
        df[column].fillna(df[column].median(), inplace=True)
    
    print(f"\nAfter handling zero values:")
    print(df.describe())
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y, df

if __name__ == "__main__":
    # Test the data loader
    filepath = download_dataset()
    if filepath:
        X, y, df = load_and_preprocess_data(filepath)
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
