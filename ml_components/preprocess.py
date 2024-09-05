import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def handle_preprocessing(dataset, steps):
    """Apply a series of preprocessing steps to the dataset."""
    df = pd.DataFrame(dataset)
    
    for step in steps:
        if step == 'scale':
            df = scale_data(df)
        elif step == 'normalize':
            df = normalize_data(df)
        elif step == 'encode':
            df = encode_data(df)
        elif step == 'fill_missing':
            df = fill_missing_values(df)
       
    
    return df.to_dict()

def scale_data(df):
    """Scale the dataset using Standard Scaler."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def normalize_data(df):
    """Normalize the dataset using MinMaxScaler."""
    normalizer = MinMaxScaler()
    normalized_data = normalizer.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)

def encode_data(df):
    """Encode categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.get_dummies(df, drop_first=True)
    return encoded_df

def fill_missing_values(df, strategy='mean'):
    """Handle missing values using SimpleImputer."""
    imputer = SimpleImputer(strategy=strategy)
    filled_data = imputer.fit_transform(df)
    return pd.DataFrame(filled_data, columns=df.columns)

def handle_outliers(df):
   
    return df

def data_split(dataset, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X, y = dataset['features'], dataset['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def custom_preprocessing_pipeline(df, steps):
    """Apply a custom preprocessing pipeline defined by the user."""
    for step in steps:
        if step == 'custom_step_1':
            df = custom_step_1(df)
        elif step == 'custom_step_2':
            df = custom_step_2(df)
       
    return df
