# **House Price Prediction - Regression Model with MLOps**
---
# **Overview :**
This project involves building a regression model to predict house prices and deploying it using MLOps practices. It includes comprehensive data preprocessing, feature engineering, model selection, and performance evaluation, followed by version control, experiment tracking, containerization, and deployment.

---

# **Problem Statement :**
Predict the SalePrice of residential homes in Ames, Iowa based on various numerical and categorical features. The model is trained on a labeled dataset and tested on an unlabeled dataset.

---

# **Key Steps :**

## 1. Data Loading and Inspection
Loaded both training and test datasets.

Checked missing values, data types, and overall structure.

## 2. Exploratory Data Analysis (EDA)
Univariate and bivariate analysis

Identified numerical and categorical variables.

Analyzed feature correlation with target variable.

## 3. Feature Engineering
Handling missing values via imputation (mean/median/mode).

Feature transformation for skewed distributions.

Created new features like total area, age of house, etc.

## 4. Data Preprocessing
Encoding categorical variables using Label encoding.

Scaling numerical features using StandardScaler, MinMaxScaler, and RobustScaler.

Ensured the same preprocessing applied to both train and test sets.

## 5. Model Building
Tested multiple regression models:

Linear Regression

Random Forest Regressor

XGBoost Regressor


## 6. Model Evaluation
Models were evaluated on:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared (R²)

The best model was selected based on its performance metrics and generalizability.

--- 

# **Results :**
The model performed well on the training data and was able to generalize effectively. XGBoost and Random Forest showed superior performance compared to baseline models. Evaluation metrics and feature importance plots were used to understand and interpret model predictions.

---

## **Technologies Used :**
Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

XGBoost

Jupyter Notebook

# **Project By** 
### **Anirudha Pradip Johare**
--- 

## **Contact**
### **Email :** anirudhajohare@gmail.com
### **Linkedin :** https://www.linkedin.com/in/anirudhajohare/
