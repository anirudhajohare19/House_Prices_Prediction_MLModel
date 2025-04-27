from src.House_prices_Prediction_Model.components.data_ingestion import load_data
from src.House_prices_Prediction_Model.components.data_transformation import preprocess_data, split_data
from src.House_prices_Prediction_Model.components.model_trainer import train_model
from src.House_prices_Prediction_Model.components.model_evaluation import evaluate_model


def main():
    # Step 1: Load Data
    train_df, test_df = load_data()
    
    # Step 2: Preprocess Data
    X_train, X_test, y_train = preprocess_data(train_df, test_df)
    
    # Step 3: Train-Test Split
    X_train_split, X_val_split, y_train_split, y_val_split = split_data(X_train, y_train)
    
    # Step 4: Train Model
    model = train_model(X_train_split, y_train_split)
    
    # Step 5: Evaluate Model
    evaluate_model(model, X_val_split, y_val_split)


if __name__ == "__main__":
    main()
