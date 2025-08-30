import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class CLVPipeline:
    """
    Complete CLV prediction and customer segmentation pipeline
    Updated for your exact column names: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_columns = None
        self.results = {}
        
    def get_default_config(self):
        """
        Default configuration for the pipeline
        """
        return {
            'test_size': 0.2,
            'random_state': 42,
            'n_features': 5,
            'xgb_params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0
            },
            'segment_thresholds': {
                'high_percentile': 0.8,  # Top 20%
                'low_percentile': 0.3    # Bottom 30%
            }
        }
    
    def step1_load_and_preprocess_data(self, file_path):
        """
        Step 1: Load and preprocess data with your exact column names
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        
        print("Loading Online Retail dataset...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Verify expected columns are present
        expected_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        # Remove cancelled transactions (Invoice starting with 'C')
        df = df[~df['Invoice'].astype(str).str.startswith('C')]
        print(f"After removing cancelled transactions: {df.shape}")
        
        # Remove null Customer IDs
        df = df.dropna(subset=['Customer ID'])
        print(f"After removing null Customer IDs: {df.shape}")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Calculate total amount per transaction
        df['TotalAmount'] = df['Quantity'] * df['Price']
        
        # Remove negative quantities and prices
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        print(f"After removing negative values: {df.shape}")
        
        # Store raw data
        self.raw_data = df
        
        return df
    
    def step2_aggregate_customer_data(self, df):
        """
        Step 2: Aggregate purchase data per customer using your column names
        """
        print("\nSTEP 2: CUSTOMER DATA AGGREGATION")
        print("-" * 40)
        
        current_date = df['InvoiceDate'].max()
        
        # Customer-level aggregations using your exact column names
        customer_data = df.groupby('Customer ID').agg({
            'TotalAmount': ['sum', 'mean', 'count'],
            'Invoice': 'nunique',
            'InvoiceDate': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_data.columns = [
            'CustomerID', 'TotalPurchaseAmount', 'AvgOrderValue', 
            'TotalTransactions', 'NumberOfOrders', 'FirstPurchaseDate', 'LastPurchaseDate'
        ]
        
        # Calculate recency and tenure
        customer_data['Recency'] = (current_date - customer_data['LastPurchaseDate']).dt.days
        customer_data['Tenure'] = (customer_data['LastPurchaseDate'] - customer_data['FirstPurchaseDate']).dt.days + 1
        customer_data['Frequency'] = customer_data['NumberOfOrders'] / customer_data['Tenure']
        customer_data['Frequency'] = customer_data['Frequency'].fillna(0)
        
        # Drop date columns
        customer_data = customer_data.drop(['FirstPurchaseDate', 'LastPurchaseDate'], axis=1)
        
        print(f"Customer-level dataset shape: {customer_data.shape}")
        
        # Store aggregated data
        self.customer_data = customer_data
        
        return customer_data
    
    def step3_engineer_clv_features(self, df):
        """
        Step 3: Engineer CLV-related features and create target variable
        """
        print("\nSTEP 3: FEATURE ENGINEERING")
        print("-" * 40)
        
        # Create CLV target variable
        df['RecencyAdjusted'] = df['Recency'].replace(0, 1)
        df['CLV'] = df['AvgOrderValue'] * df['NumberOfOrders'] * (df['Tenure'] / df['RecencyAdjusted'])
        df['CLV'] = df['CLV'].replace([np.inf, -np.inf], df['CLV'].median())
        
        # Additional features
        df['AvgDaysBetweenOrders'] = df['Tenure'] / df['NumberOfOrders']
        df['TotalOrdersPerTenure'] = df['NumberOfOrders'] / (df['Tenure'] / 365.25)  # Orders per year
        df['MonetaryValue'] = df['TotalPurchaseAmount']
        
        print("Features engineered:")
        feature_list = ['TotalPurchaseAmount', 'AvgOrderValue', 'TotalTransactions', 
                       'NumberOfOrders', 'Recency', 'Tenure', 'Frequency', 
                       'AvgDaysBetweenOrders', 'TotalOrdersPerTenure', 'MonetaryValue']
        for feature in feature_list:
            print(f"  • {feature}")
        
        print(f"\nCLV Statistics:")
        print(df['CLV'].describe())
        
        return df
    
    def step4_automated_feature_selection(self, df):
        """
        Step 4: Perform automated feature selection
        """
        print("\nSTEP 4: AUTOMATED FEATURE SELECTION")
        print("-" * 40)
        
        # Define feature columns (excluding target and ID)
        feature_cols = ['TotalPurchaseAmount', 'AvgOrderValue', 'TotalTransactions',
                       'NumberOfOrders', 'Recency', 'Tenure', 'Frequency',
                       'AvgDaysBetweenOrders', 'TotalOrdersPerTenure', 'MonetaryValue']
        
        X = df[feature_cols]
        y = df['CLV']
        
        # Apply SelectKBest
        selector = SelectKBest(score_func=f_regression, k=self.config['n_features'])
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        feature_scores = selector.scores_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Score': feature_scores,
            'Selected': selector.get_support()
        }).sort_values('Score', ascending=False)
        
        print(f"Top {self.config['n_features']} selected features:")
        for feature in selected_features:
            score = importance_df[importance_df['Feature'] == feature]['Score'].iloc[0]
            print(f"  • {feature} (Score: {score:.2f})")
        
        # Store results
        self.feature_selector = selector
        self.feature_columns = selected_features
        self.results['feature_importance'] = importance_df
        
        return selected_features, importance_df
    
    def step5_train_xgboost_model(self, df):
        """
        Step 5: Train XGBoost regressor and save the model
        """
        print("\nSTEP 5: MODEL TRAINING")
        print("-" * 40)
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['CLV']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(**self.config['xgb_params'])
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        print(f"Model Performance:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model and components
        joblib.dump(self.model, 'models/clv_model_pipeline.pkl')
        joblib.dump(self.feature_columns, 'models/feature_columns_pipeline.pkl')
        
        print("Model saved successfully!")
        
        # Store results
        self.results['model_performance'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
        
        return self.model
    
    def step6_load_new_customer_data(self, file_path):
        """
        Step 6: Load new customer data for prediction
        """
        print("\nSTEP 6: LOADING NEW CUSTOMER DATA")
        print("-" * 40)
        
        try:
            new_data = pd.read_csv(file_path)
            print(f"Loaded new customer data: {new_data.shape}")
        except FileNotFoundError:
            print("New customer data file not found. Creating sample data...")
            new_data = self.create_sample_customer_data()
        
        # Store new data
        self.new_customer_data = new_data
        
        return new_data
    
    def step7_predict_clv(self, new_data):
        """
        Step 7: Predict CLV for new customers
        """
        print("\nSTEP 7: CLV PREDICTION")
        print("-" * 40)
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")
        
        # Select features and predict
        X_new = new_data[self.feature_columns]
        predictions = self.model.predict(X_new)
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'CustomerID': new_data.get('CustomerID', range(len(predictions))),
            'Predicted_CLV': predictions
        })
        
        print(f"Predictions generated for {len(predictions)} customers")
        print(f"Average predicted CLV: ${predictions.mean():.2f}")
        print(f"CLV range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        # Store predictions
        self.predictions = prediction_df
        
        return prediction_df
    
    def step8_assign_customer_segments(self, prediction_df):
        """
        Step 8: Assign customer segments based on CLV with improved visualization
        """
        print("\nSTEP 8: CUSTOMER SEGMENTATION")
        print("-" * 40)
        
        # Calculate thresholds
        high_threshold = prediction_df['Predicted_CLV'].quantile(
            self.config['segment_thresholds']['high_percentile']
        )
        low_threshold = prediction_df['Predicted_CLV'].quantile(
            self.config['segment_thresholds']['low_percentile']
        )
        
        # Assign segments
        def assign_segment(clv_value):
            if clv_value >= high_threshold:
                return 'High Value'
            elif clv_value >= low_threshold:
                return 'Medium Value'
            else:
                return 'Low Value'
        
        prediction_df['CLV_Segment'] = prediction_df['Predicted_CLV'].apply(assign_segment)
        
        # Display segment distribution
        segment_counts = prediction_df['CLV_Segment'].value_counts()
        print("Segment Distribution:")
        for segment, count in segment_counts.items():
            percentage = (count / len(prediction_df)) * 100
            avg_clv = prediction_df[prediction_df['CLV_Segment'] == segment]['Predicted_CLV'].mean()
            print(f"  • {segment}: {count} customers ({percentage:.1f}%) - Avg CLV: ${avg_clv:.2f}")
        
        # Create improved visualization
        self.create_improved_segmentation_plots(prediction_df)
        
        # Store segmented data
        self.segmented_data = prediction_df
        
        return prediction_df
    
    def create_improved_segmentation_plots(self, df):
        """
        Create improved customer segmentation visualizations with better formatting
        """
        print("Creating improved segmentation visualizations...")
        
        # Create output directory
        os.makedirs('outputs/plots', exist_ok=True)
        
        # Set up the plotting style with larger figure
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Color palette for consistency
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
        
        # 1. Distribution of CLV by segment (box plot)
        sns.boxplot(data=df, x='CLV_Segment', y='Predicted_CLV', 
                   palette=colors, ax=axes[0,0])
        axes[0,0].set_title('CLV Distribution by Customer Segment', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Predicted CLV ($)', fontsize=12)
        axes[0,0].set_xlabel('Customer Segment', fontsize=12)
        axes[0,0].tick_params(axis='x', rotation=0, labelsize=11)
        
        # 2. Count of customers by segment
        segment_counts = df['CLV_Segment'].value_counts()
        bars = axes[0,1].bar(segment_counts.index, segment_counts.values, color=colors[:len(segment_counts)])
        axes[0,1].set_title('Number of Customers by Segment', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Number of Customers', fontsize=12)
        axes[0,1].set_xlabel('Customer Segment', fontsize=12)
        axes[0,1].tick_params(axis='x', rotation=0, labelsize=11)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 3. CLV histogram with segment colors - IMPROVED VERSION
        bin_edges = np.linspace(df['Predicted_CLV'].min(), df['Predicted_CLV'].max(), 31)
        
        for i, segment in enumerate(['High Value', 'Medium Value', 'Low Value']):
            if segment in df['CLV_Segment'].values:
                segment_data = df[df['CLV_Segment'] == segment]['Predicted_CLV']
                axes[1,0].hist(segment_data, bins=bin_edges, alpha=0.7, 
                              label=segment, color=colors[i], edgecolor='black', linewidth=0.5)
        
        axes[1,0].set_xlabel('Predicted CLV ($)', fontsize=12)
        axes[1,0].set_ylabel('Number of Customers', fontsize=12)
        axes[1,0].set_title('CLV Distribution by Segment (Histogram)', fontsize=14, fontweight='bold')
        
        # Improved legend positioning and formatting
        legend = axes[1,0].legend(loc='upper right', frameon=True, fancybox=True, 
                                 shadow=True, fontsize=11)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 4. Pie chart of segment distribution - IMPROVED VERSION
        segment_counts = df['CLV_Segment'].value_counts()
        
        # Calculate percentages
        percentages = (segment_counts.values / segment_counts.sum()) * 100
        
        # Create custom labels with both count and percentage
        labels = []
        for segment, count, pct in zip(segment_counts.index, segment_counts.values, percentages):
            labels.append(f'{segment}\n{count} customers\n({pct:.1f}%)')
        
        wedges, texts, autotexts = axes[1,1].pie(
            segment_counts.values, 
            labels=labels,
            colors=colors[:len(segment_counts)],
            autopct='',  # We'll add custom labels
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        axes[1,1].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # Make the pie chart circular
        axes[1,1].axis('equal')
        
        # Adjust spacing between subplots
        plt.tight_layout(pad=3.0)
        
        # Save the plot
        output_path = 'outputs/plots/customer_segmentation_analysis_improved.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Improved visualization saved to: {output_path}")
        
        plt.show()
    
    def step9_export_final_results(self, segmented_data):
        """
        Step 9: Export final results to CSV report
        """
        print("\nSTEP 9: EXPORTING RESULTS")
        print("-" * 40)
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Main results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'outputs/clv_analysis_results_{timestamp}.csv'
        segmented_data.to_csv(output_file, index=False)
        
        # Summary report
        summary_file = f'outputs/clv_summary_report_{timestamp}.csv'
        
        # Create summary statistics
        summary_stats = segmented_data.groupby('CLV_Segment')['Predicted_CLV'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', 'sum'
        ]).round(2)
        
        summary_stats.to_csv(summary_file)
        
        print(f"Results exported to:")
        print(f"  • {output_file} (detailed results)")
        print(f"  • {summary_file} (summary statistics)")
        
        # Store file names
        self.output_files = {
            'detailed_results': output_file,
            'summary_report': summary_file
        }
        
        return output_file, summary_file
    
    def create_sample_customer_data(self, n_customers=200):
        """
        Create sample customer data for demonstration
        """
        np.random.seed(self.config['random_state'])
        
        sample_data = pd.DataFrame({
            'CustomerID': range(20000, 20000 + n_customers),
            'TotalPurchaseAmount': np.random.gamma(2, 300, n_customers),
            'AvgOrderValue': np.random.gamma(2, 75, n_customers),
            'TotalTransactions': np.random.poisson(8, n_customers) + 1,
            'NumberOfOrders': np.random.poisson(6, n_customers) + 1,
            'Recency': np.random.exponential(45, n_customers),
            'Tenure': np.random.gamma(2, 100, n_customers) + 30,
            'Frequency': np.random.gamma(1, 0.05, n_customers),
            'AvgDaysBetweenOrders': np.random.gamma(2, 20, n_customers),
            'TotalOrdersPerTenure': np.random.gamma(1, 2, n_customers),
            'MonetaryValue': np.random.gamma(2, 300, n_customers)
        })
        
        # Ensure positive values
        for col in sample_data.columns:
            if col != 'CustomerID':
                sample_data[col] = np.abs(sample_data[col])
        
        # Save sample data
        sample_file = 'sample_new_customers_pipeline.csv'
        sample_data.to_csv(sample_file, index=False)
        print(f"Sample customer data created: {sample_file}")
        
        return sample_data
    
    def run_complete_pipeline(self, raw_data_path, new_data_path=None):
        """
        Run the complete CLV prediction pipeline
        """
        print("🚀 STARTING COMPLETE CLV PREDICTION PIPELINE")
        print("=" * 80)
        
        try:
            # Step 1: Load and preprocess data
            raw_df = self.step1_load_and_preprocess_data(raw_data_path)
            
            # Step 2: Aggregate customer data
            customer_df = self.step2_aggregate_customer_data(raw_df)
            
            # Step 3: Engineer features
            featured_df = self.step3_engineer_clv_features(customer_df)
            
            # Step 4: Feature selection
            selected_features, importance_df = self.step4_automated_feature_selection(featured_df)
            
            # Step 5: Train model
            model = self.step5_train_xgboost_model(featured_df)
            
            # Step 6: Load new customer data
            if new_data_path:
                new_data = self.step6_load_new_customer_data(new_data_path)
            else:
                new_data = self.create_sample_customer_data()
            
            # Step 7: Predict CLV
            predictions = self.step7_predict_clv(new_data)
            
            # Step 8: Assign segments with improved visualization
            segmented_predictions = self.step8_assign_customer_segments(predictions)
            
            # Step 9: Export results
            output_file, summary_file = self.step9_export_final_results(segmented_predictions)
            
            # Generate final summary
            self.generate_pipeline_summary()
            
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            return segmented_predictions
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            raise e
    
    def generate_pipeline_summary(self):
        """
        Generate a comprehensive summary of the pipeline results
        """
        print("\n📊 PIPELINE SUMMARY")
        print("=" * 60)
        
        # Data summary
        if hasattr(self, 'customer_data'):
            print(f"Customers processed: {len(self.customer_data):,}")
        
        # Model performance
        if 'model_performance' in self.results:
            perf = self.results['model_performance']
            print(f"Model Test R²: {perf['test_r2']:.4f}")
            print(f"Model Test RMSE: ${perf['test_rmse']:,.2f}")
        
        # Feature importance
        if 'feature_importance' in self.results:
            print(f"\nTop 3 Most Important Features:")
            top_features = self.results['feature_importance'].head(3)
            for _, row in top_features.iterrows():
                print(f"  • {row['Feature']}: {row['Score']:.2f}")
        
        # Prediction summary
        if hasattr(self, 'predictions'):
            pred_stats = self.predictions['Predicted_CLV']
            print(f"\nPrediction Summary:")
            print(f"  Total customers predicted: {len(pred_stats):,}")
            print(f"  Average CLV: ${pred_stats.mean():,.2f}")
            print(f"  Median CLV: ${pred_stats.median():,.2f}")
            print(f"  CLV Range: ${pred_stats.min():,.2f} - ${pred_stats.max():,.2f}")
        
        # Segment distribution
        if hasattr(self, 'segmented_data'):
            print(f"\nSegment Distribution:")
            segment_summary = self.segmented_data.groupby('CLV_Segment').agg({
                'CustomerID': 'count',
                'Predicted_CLV': ['mean', 'sum']
            }).round(2)
            
            for segment in ['High Value', 'Medium Value', 'Low Value']:
                if segment in segment_summary.index:
                    count = segment_summary.loc[segment, ('CustomerID', 'count')]
                    avg_clv = segment_summary.loc[segment, ('Predicted_CLV', 'mean')]
                    total_clv = segment_summary.loc[segment, ('Predicted_CLV', 'sum')]
                    print(f"  • {segment}: {count} customers, Avg: ${avg_clv:,.2f}, Total: ${total_clv:,.2f}")
        
        # Files created
        print(f"\nFiles Created:")
        print(f"  • models/clv_model_pipeline.pkl (trained model)")
        print(f"  • models/feature_columns_pipeline.pkl (feature list)")
        print(f"  • outputs/plots/customer_segmentation_analysis_improved.png (improved charts)")
        if hasattr(self, 'output_files'):
            for desc, filename in self.output_files.items():
                print(f"  • {filename} ({desc})")

# Convenience functions for different use cases
def run_full_pipeline(raw_data_path, new_data_path=None, config=None):
    """
    Run the complete pipeline with default settings
    """
    pipeline = CLVPipeline(config)
    return pipeline.run_complete_pipeline(raw_data_path, new_data_path)

def batch_predict_clv(model_path, new_data_path, output_path):
    """
    Perform batch CLV prediction using a saved model
    """
    pipeline = CLVPipeline()
    return pipeline.load_and_predict_batch(model_path, new_data_path, output_path)

def main():
    """
    Main function demonstrating the complete pipeline
    """
    print("🎯 CLV PREDICTION AUTOMATION SCRIPT - UPDATED FOR YOUR DATA")
    print("=" * 80)
    print("Column format: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country")
    print("=" * 80)
    
    # Configuration
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'n_features': 5,
        'xgb_params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        },
        'segment_thresholds': {
            'high_percentile': 0.8,
            'low_percentile': 0.3
        }
    }
    
    # Initialize pipeline
    pipeline = CLVPipeline(config)
    
    # Example usage
    try:
        # Option 1: Run complete pipeline
        raw_data_path = 'online_retail.csv'  # Your data file
        results = pipeline.run_complete_pipeline(raw_data_path)
        
        print("\n🎉 All tasks completed successfully!")
        print("Check the outputs/ folder for improved visualizations and results.")
        
    except FileNotFoundError:
        print("\n⚠️  online_retail.csv file not found.")
        print("Please ensure the file exists in the current directory.")
        
        # Demonstrate with sample data
        print("\nCreating sample data for demonstration...")
        sample_data = pipeline.create_sample_customer_data(500)
        
        # Create mock processed data for demonstration
        processed_sample = pipeline.step3_engineer_clv_features(sample_data)
        selected_features, _ = pipeline.step4_automated_feature_selection(processed_sample)
        model = pipeline.step5_train_xgboost_model(processed_sample)
        
        # Predict on new sample
        new_sample = pipeline.create_sample_customer_data(100)
        predictions = pipeline.step7_predict_clv(new_sample)
        final_results = pipeline.step8_assign_customer_segments(predictions)
        pipeline.step9_export_final_results(final_results)
        
        print("\n✅ Demo completed with sample data!")

if __name__ == "__main__":
    main()