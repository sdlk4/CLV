import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def train_and_save_xgboost_model():
    """
    Train XGBoost model and save it using joblib
    """
    print("Training and saving XGBoost CLV model...")
    
    # Load data
    df = pd.read_csv('customer_data_with_clv.csv')
    
    # Prepare features and target
    feature_cols = ['TotalPurchaseAmount', 'AvgOrderValue', 'NumberOfOrders', 'Recency', 'Frequency']
    X = df[feature_cols]
    y = df['CLV']
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model Training R² Score: {train_score:.4f}")
    print(f"Model Test R² Score: {test_score:.4f}")
    
    # Save the model
    joblib.dump(model, 'clv_xgboost_model.pkl')
    print("Model saved as 'clv_xgboost_model.pkl'")
    
    # Save feature columns for future use
    joblib.dump(feature_cols, 'feature_columns.pkl')
    print("Feature columns saved as 'feature_columns.pkl'")
    
    return model, feature_cols

def load_model_and_predict(new_data_path):
    """
    Load the trained model and predict CLV for new customer data
    """
    print("\nLoading trained model and making predictions...")
    
    # Load the saved model and feature columns
    try:
        model = joblib.load('clv_xgboost_model.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        print("Model and feature columns loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please run training first.")
        return None
    
    print(f"Expected features: {feature_cols}")
    
    # Load new customer data
    try:
        new_data = pd.read_csv(new_data_path)
        print(f"New data shape: {new_data.shape}")
        print(f"New data columns: {list(new_data.columns)}")
    except FileNotFoundError:
        print(f"New data file '{new_data_path}' not found. Creating sample data...")
        new_data = create_sample_new_data()
    
    # Ensure all required features are present
    missing_features = set(feature_cols) - set(new_data.columns)
    if missing_features:
        print(f"Missing features in new data: {missing_features}")
        print("Please ensure new data contains all required features.")
        return None
    
    # Select features for prediction
    X_new = new_data[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'CustomerID': new_data['CustomerID'] if 'CustomerID' in new_data.columns else range(len(predictions)),
        'Predicted_CLV': predictions
    })
    
    # Save predictions
    results_df.to_csv('clv_predictions.csv', index=False)
    print(f"Predictions saved to 'clv_predictions.csv'")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Average predicted CLV: ${predictions.mean():.2f}")
    print(f"CLV range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    
    return results_df

def create_sample_new_data(n_customers=100):
    """
    Create sample new customer data for demonstration
    """
    print("Creating sample new customer data...")
    
    np.random.seed(42)
    
    # Generate realistic customer data
    sample_data = pd.DataFrame({
        'CustomerID': range(10000, 10000 + n_customers),
        'TotalPurchaseAmount': np.random.gamma(2, 200, n_customers),
        'AvgOrderValue': np.random.gamma(2, 50, n_customers),
        'NumberOfOrders': np.random.poisson(5, n_customers) + 1,
        'Recency': np.random.exponential(30, n_customers),
        'Frequency': np.random.gamma(1, 0.1, n_customers)
    })
    
    # Ensure positive values
    sample_data['TotalPurchaseAmount'] = np.abs(sample_data['TotalPurchaseAmount'])
    sample_data['AvgOrderValue'] = np.abs(sample_data['AvgOrderValue'])
    sample_data['Recency'] = np.abs(sample_data['Recency'])
    sample_data['Frequency'] = np.abs(sample_data['Frequency'])
    
    # Save sample data
    sample_data.to_csv('sample_new_customers.csv', index=False)
    print("Sample data created and saved as 'sample_new_customers.csv'")
    
    return sample_data

def batch_predict_from_model(model_path, data_path, output_path):
    """
    Utility function for batch prediction
    """
    # Load model
    model = joblib.load(model_path)
    feature_cols = joblib.load('feature_columns.pkl')
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Predict
    X = data[feature_cols]
    predictions = model.predict(X)
    
    # Create results
    results = pd.DataFrame({
        'CustomerID': data['CustomerID'] if 'CustomerID' in data.columns else range(len(predictions)),
        'Predicted_CLV': predictions
    })
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"Batch predictions saved to '{output_path}'")
    
    return results

def validate_model_performance():
    """
    Validate the saved model performance
    """
    print("\nValidating saved model performance...")
    
    # Load original data
    df = pd.read_csv('customer_data_with_clv.csv')
    feature_cols = ['TotalPurchaseAmount', 'AvgOrderValue', 'NumberOfOrders', 'Recency', 'Frequency']
    
    X = df[feature_cols]
    y = df['CLV']
    
    # Load saved model
    model = joblib.load('clv_xgboost_model.pkl')
    
    # Make predictions on full dataset
    predictions = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Saved model validation metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main():
    """
    Main function to train, save model and make predictions
    """
    # Step 1: Train and save XGBoost model
    model, feature_cols = train_and_save_xgboost_model()
    
    # Step 2: Validate saved model
    validation_metrics = validate_model_performance()
    
    # Step 3: Load model and predict on new data
    # Try to load existing new data, otherwise create sample data
    new_data_path = 'sample_new_customers.csv'
    predictions_df = load_model_and_predict(new_data_path)
    
    if predictions_df is not None:
        print("\n=== Prediction Summary ===")
        print(predictions_df.head(10))
        print(f"\nTotal customers predicted: {len(predictions_df)}")
        print(f"Prediction statistics:")
        print(predictions_df['Predicted_CLV'].describe())
    
    print("\n=== Process Complete ===")
    print("Files created:")
    print("- clv_xgboost_model.pkl (trained model)")
    print("- feature_columns.pkl (feature list)")
    print("- sample_new_customers.csv (sample data)")
    print("- clv_predictions.csv (predictions)")

if __name__ == "__main__":
    main()