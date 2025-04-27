import pandas as pd

def load_data():
    D1 = 'https://raw.githubusercontent.com/anirudhajohare19/House_Prices_Prediction_MLModel/refs/heads/main/research/train.csv'
    D2 = 'https://raw.githubusercontent.com/anirudhajohare19/House_Prices_Prediction_MLModel/refs/heads/main/research/test.csv'
    
    train_df = pd.read_csv(D1)
    test_df = pd.read_csv(D2)
    
    return train_df, test_df
