import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data_with_clv(file_path):
    """
    Load customer data with CLV
    """
    df = pd.read_csv(file_path)
    print(f"Loaded dataset shape: {df.shape}")
    return df

def prepare_features_target(df):
    """
    Prepare features and target variable for model training
    """
    # Use top features identified from feature selection
    feature_cols = ['TotalPurchaseAmount', 'AvgOrderValue', 'NumberOfOrders', 'Recency', 'Frequency']
    
    X = df[feature_cols]
    y = df['CLV']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Features used: {feature_cols}")
    
    return X, y, feature_cols

def train_evaluate_models(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate three different models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
    }
    
    results = {}
    trained_models = {}
    
    print("\n=== Model Training and Evaluation ===")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Linear Regression, original data for tree-based models
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        if name == 'Linear Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Predictions': y_pred
        }
        
        trained_models[name] = model
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results, trained_models, scaler, X_test, y_test

def plot_model_comparison(results):
    """
    Create visualizations comparing model performance
    """
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R²']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE comparison
    sns.barplot(data=comparison_df, x='Model', y='RMSE', ax=axes[0])
    axes[0].set_title('Root Mean Squared Error (RMSE)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    sns.barplot(data=comparison_df, x='Model', y='MAE', ax=axes[1])
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    sns.barplot(data=comparison_df, x='Model', y='R²', ax=axes[2])
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def plot_predictions_vs_actual(results, y_test):
    """
    Plot predictions vs actual values for each model
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['Predictions']
        
        axes[idx].scatter(y_test, y_pred, alpha=0.6)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual CLV')
        axes[idx].set_ylabel('Predicted CLV')
        axes[idx].set_title(f'{model_name}\nR² = {metrics["R²"]:.4f}')
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(trained_models, feature_cols):
    """
    Analyze feature importance for tree-based models
    """
    print("\n=== Feature Importance Analysis ===")
    
    # Random Forest feature importance
    if 'Random Forest' in trained_models:
        rf_importance = trained_models['Random Forest'].feature_importances_
        rf_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(rf_importance_df)
    
    # XGBoost feature importance
    if 'XGBoost' in trained_models:
        xgb_importance = trained_models['XGBoost'].feature_importances_
        xgb_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': xgb_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nXGBoost Feature Importance:")
        print(xgb_importance_df)
        
        # Plot feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.barplot(data=rf_importance_df, x='Importance', y='Feature', ax=ax1)
        ax1.set_title('Random Forest Feature Importance')
        
        sns.barplot(data=xgb_importance_df, x='Importance', y='Feature', ax=ax2)
        ax2.set_title('XGBoost Feature Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance_models.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Load data
    df = load_data_with_clv('customer_data_with_clv.csv')
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_target(df)
    
    # Train and evaluate models
    results, trained_models, scaler, X_test, y_test = train_evaluate_models(X, y)
    
    # Create comparison plots
    comparison_df = plot_model_comparison(results)
    plot_predictions_vs_actual(results, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(trained_models, feature_cols)
    
    # Print summary
    print("\n=== Model Performance Summary ===")
    print(comparison_df.round(4))
    
    # Identify best model
    best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
    print(f"\nBest performing model: {best_model_name}")
    
    # Save results
    comparison_df.to_csv('model_performance_comparison.csv', index=False)
    print("Model comparison results saved to 'model_performance_comparison.csv'")
    
    return results, trained_models, scaler, feature_cols, best_model_name

if __name__ == "__main__":
    results, trained_models, scaler, feature_cols, best_model = main()