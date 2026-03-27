import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the raw dataframe into training and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Exited'])
    return train_df, val_df

def get_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns, excluding 'Surname' and the target 'Exited'.
    """
    # Drop non-predictive or high-cardinality columns as requested
    cols_to_drop = ['Surname', 'Exited', 'CustomerId', 'id', 'RowNumber']
    input_cols = [c for c in df.columns if c not in cols_to_drop]
    
    numeric_cols = df[input_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df[input_cols].select_dtypes(include='object').columns.tolist()
    
    return numeric_cols, categorical_cols

def scale_numeric_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fits a MinMaxScaler on training data and transforms both train and validation sets.
    """
    scaler = MinMaxScaler().fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_categorical_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Applies One-Hot Encoding to categorical columns and returns the encoder and new column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    
    return train_df, val_df, encoder, encoded_cols

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, Any]:
    """
    Main pipeline to process bank churn data. Returns split datasets, targets, 
    feature names, and fitted transformers.
    """
    target_col = 'Exited'
    
    # 1. Split data
    train_df, val_df = split_data(raw_df)
    
    # 2. Identify columns
    numeric_cols, categorical_cols = get_feature_cols(train_df)
    
    # 3. Scale (Optional)
    scaler = None
    if scale_numeric:
        train_df, val_df, scaler = scale_numeric_features(train_df, val_df, numeric_cols)
    
    # 4. Encode
    train_df, val_df, encoder, encoded_cols = encode_categorical_features(train_df, val_df, categorical_cols)
    
    # Define final input columns (numeric + one-hot columns)
    input_cols = numeric_cols + encoded_cols
    
    return {
        'X_train': train_df[input_cols],
        'train_targets': train_df[target_col],
        'X_val': val_df[input_cols],
        'val_targets': val_df[target_col],
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(
    new_df: pd.DataFrame, 
    input_cols: List[str], 
    scaler: Optional[MinMaxScaler], 
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Processes new data (e.g. test.csv) using pre-trained scaler and encoder.
    """
    df = new_df.copy()
    
    # Get original feature lists to know what to transform
    numeric_cols, categorical_cols = get_feature_cols(df)
    
    # Apply Scaling if scaler was provided
    if scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Apply Encoding
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    
    return df[input_cols]
