import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_df, test_df):
    # Dropping columns
    train_df = train_df.drop(['PoolQC', "Alley", "Fence", 'MiscFeature'], axis=1)
    test_df = test_df.drop(['PoolQC', "Alley", "Fence", 'MiscFeature'], axis=1)
    
    # Filling None values
    none_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    
    for col in none_cols:
        train_df[col] = train_df[col].fillna("None")
        test_df[col] = test_df[col].fillna("None")
    
    # Filling 0 values
    zero_fill_cols = ['GarageYrBlt', 'MasVnrArea']
    for col in zero_fill_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    
    # Median Imputation
    train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].median())
    test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].median())
    
    # Label Encoding
    combined = pd.concat([train_df.drop('SalePrice', axis=1), test_df], axis=0)
    cat_cols = combined.select_dtypes(include='object').columns

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
        label_encoders[col] = le

    # Split back
    X_train = combined.iloc[:train_df.shape[0], :]
    X_test = combined.iloc[train_df.shape[0]:, :]
    y_train = train_df['SalePrice']
    
    return X_train, X_test, y_train

def split_data(X, y):
    return train_test_split(X, y, test_size=0.30, random_state=42)
