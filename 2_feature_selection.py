import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_customer_data(file_path):
    """
    Load preprocessed customer data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded customer data shape: {df.shape}")
    return df

def create_clv_target(df):
    """
    Create CLV target variable using a simple formula
    CLV = Average Order Value × Number of Orders × (Tenure/Recency)
    """
    # Avoid division by zero for recency
    df['RecencyAdjusted'] = df['Recency'].replace(0, 1)
    
    # Calculate CLV
    df['CLV'] = df['AvgOrderValue'] * df['NumberOfOrders'] * (df['Tenure'] / df['RecencyAdjusted'])
    
    # Handle infinite values
    df['CLV'] = df['CLV'].replace([np.inf, -np.inf], df['CLV'].median())
    
    print(f"CLV Statistics:")
    print(df['CLV'].describe())
    
    return df

def apply_feature_selection(df, target_col='CLV', k_best=5):
    """
    Apply automated feature selection using SelectKBest and RFE
    """
    # Prepare features and target
    feature_cols = ['TotalPurchaseAmount', 'AvgOrderValue', 'TotalTransactions', 
                   'NumberOfOrders', 'Recency', 'Tenure', 'Frequency']
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_cols}")
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Method 1: SelectKBest with f_regression
    print("\n=== SelectKBest with f_regression ===")
    selector_kbest = SelectKBest(score_func=f_regression, k=k_best)
    X_kbest = selector_kbest.fit_transform(X_scaled, y)
    
    # Get selected features
    selected_features_kbest = [feature_cols[i] for i in selector_kbest.get_support(indices=True)]
    feature_scores = selector_kbest.scores_
    
    print(f"Selected features ({k_best}): {selected_features_kbest}")
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Score': feature_scores,
        'Selected': selector_kbest.get_support()
    }).sort_values('Score', ascending=False)
    
    print("\nFeature Scores (f_regression):")
    print(importance_df)
    
    # Method 2: Recursive Feature Elimination (RFE)
    print("\n=== Recursive Feature Elimination (RFE) ===")
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector_rfe = RFE(estimator=estimator, n_features_to_select=k_best, step=1)
    X_rfe = selector_rfe.fit_transform(X_scaled, y)
    
    # Get selected features
    selected_features_rfe = [feature_cols[i] for i in selector_rfe.get_support(indices=True)]
    feature_ranking = selector_rfe.ranking_
    
    print(f"Selected features ({k_best}): {selected_features_rfe}")
    
    # Create RFE ranking dataframe
    rfe_df = pd.DataFrame({
        'Feature': feature_cols,
        'Ranking': feature_ranking,
        'Selected': selector_rfe.get_support()
    }).sort_values('Ranking')
    
    print("\nFeature Rankings (RFE):")
    print(rfe_df)
    
    # Visualize feature importance
    plot_feature_importance(importance_df, rfe_df)
    
    # Return both selected feature sets
    return selected_features_kbest, selected_features_rfe, importance_df, rfe_df

def plot_feature_importance(importance_df, rfe_df):
    """
    Plot feature importance and rankings
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot f_regression scores
    sns.barplot(data=importance_df, x='Score', y='Feature', 
                palette=['red' if x else 'blue' for x in importance_df['Selected']], ax=ax1)
    ax1.set_title('Feature Scores (f_regression)')
    ax1.set_xlabel('F-Score')
    
    # Plot RFE rankings
    sns.barplot(data=rfe_df, x='Ranking', y='Feature', 
                palette=['red' if x else 'blue' for x in rfe_df['Selected']], ax=ax2)
    ax2.set_title('Feature Rankings (RFE)')
    ax2.set_xlabel('Ranking (1 = Best)')
    
    plt.tight_layout()
    plt.savefig('feature_selection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_methods(selected_kbest, selected_rfe):
    """
    Compare results from both feature selection methods
    """
    print("\n=== Comparison of Feature Selection Methods ===")
    print(f"SelectKBest features: {selected_kbest}")
    print(f"RFE features: {selected_rfe}")
    
    # Find common features
    common_features = list(set(selected_kbest) & set(selected_rfe))
    unique_kbest = list(set(selected_kbest) - set(selected_rfe))
    unique_rfe = list(set(selected_rfe) - set(selected_kbest))
    
    print(f"\nCommon features: {common_features}")
    print(f"Unique to SelectKBest: {unique_kbest}")
    print(f"Unique to RFE: {unique_rfe}")
    
    return common_features

def main():
    # Load preprocessed data
    df = load_customer_data('customer_data_preprocessed.csv')
    
    # Create CLV target variable
    df = create_clv_target(df)
    
    # Apply feature selection
    selected_kbest, selected_rfe, importance_df, rfe_df = apply_feature_selection(df, k_best=5)
    
    # Compare methods
    common_features = compare_methods(selected_kbest, selected_rfe)
    
    # Save results
    results = {
        'selected_features_kbest': selected_kbest,
        'selected_features_rfe': selected_rfe,
        'common_features': common_features
    }
    
    # Save updated dataset with CLV
    df.to_csv('customer_data_with_clv.csv', index=False)
    print("\nDataset with CLV saved to 'customer_data_with_clv.csv'")
    
    # Save feature selection results
    importance_df.to_csv('feature_importance_scores.csv', index=False)
    rfe_df.to_csv('rfe_rankings.csv', index=False)
    print("Feature selection results saved to CSV files")
    
    return results

if __name__ == "__main__":
    results = main()