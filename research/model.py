# Importing Data Manipulation Libraries
import pandas as pd
import numpy as np

# Importing Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor , AdaBoostRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# importing Warnings
import warnings
warnings.filterwarnings("ignore")

# Importing Logging Libraries
import logging
logging.basicConfig(level=logging.INFO, 
                    format="[%(asctime)s]: %(levelname)s: %(message)s",
                    filename="app.log",
                    filemode="a")


# Importing the dataset

D1 = 'https://raw.githubusercontent.com/anirudhajohare19/House_Prices_Prediction_MLModel/refs/heads/main/research/train.csv'
D2 = 'https://raw.githubusercontent.com/anirudhajohare19/House_Prices_Prediction_MLModel/refs/heads/main/research/test.csv'

Train = pd.read_csv(D1)
Test = pd.read_csv(D2)

# droping the columns with missing values
# Dropped from trianing and testing dataset
Train.drop(['PoolQC',"Alley", "Fence",'MiscFeature'], axis=1, inplace=True)
Test.drop(['PoolQC',"Alley", "Fence",'MiscFeature'], axis=1, inplace=True)

# Fillinf the missing values with "None" 
# Training datset imputaition
none_cols = ['FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

for col in none_cols:
    Train[col] = Train[col].fillna("None")

# Testing datset imputaition
none_cols = ['FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

for col in none_cols:
    Test[col] = Test[col].fillna("None")

# Fill numeric columns with zero (indicating absence)
# Training datset imputaition
zero_fill_cols = ['GarageYrBlt', 'MasVnrArea']
for col in zero_fill_cols:
    Train[col] = Train[col].fillna(0)

# Testing datset imputaition
zero_fill_cols = ['GarageYrBlt', 'MasVnrArea']
for col in zero_fill_cols:
    Test[col] = Test[col].fillna(0)

# Impute with median for numeric features where missin
Train['LotFrontage'] = Train['LotFrontage'].fillna(Train['LotFrontage'].median())
Test['LotFrontage'] = Test['LotFrontage'].fillna(Test['LotFrontage'].median())

# Using Lebel Encoding
from sklearn.preprocessing import LabelEncoder

# Combine train + test to ensure consistent label encoding
combined = pd.concat([Train.drop('SalePrice', axis=1), Test], axis=0)

# Identify categorical columns
cat_cols = combined.select_dtypes(include='object').columns

# Apply label encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le


# Reassign to original train/test
X_train = combined.iloc[:Train.shape[0], :]
X_test = combined.iloc[Train.shape[0]:, :]

# Add back target column
y_train = Train['SalePrice']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.30, random_state=42)



# Model Training
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_split, y_train_split)

# Model Prediction
y_pred_xgb = xgb.predict(X_val_split)

# Evaluation Metrics
mse = mean_squared_error(y_val_split, y_pred_xgb)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val_split, y_pred_xgb)
r2 = r2_score(y_val_split, y_pred_xgb)
mape = mean_absolute_percentage_error(y_val_split, y_pred_xgb) * 100  # In percentage

# Print Results
print(f"XGBoost Evaluation Metrics:")
print(f"----------------------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")