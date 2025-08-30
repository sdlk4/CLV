import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='online_retail.csv'):
    """
    Load and preprocess the Online Retail dataset with your exact column names:
    Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    
    print(f"Loading dataset from: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"✅ Successfully loaded dataset!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Verify expected columns are present
    expected_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing expected columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print("✅ All expected columns found!")
    
    print(f"\n🧹 Starting data cleaning...")
    
    # Remove cancelled transactions (Invoice starting with 'C')
    original_len = len(df)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    cancelled_removed = original_len - len(df)
    if cancelled_removed > 0:
        print(f"✅ Removed {cancelled_removed:,} cancelled transactions")
    print(f"After removing cancelled transactions: {df.shape}")
    
    # Remove null Customer IDs
    original_len = len(df)
    df = df.dropna(subset=['Customer ID'])
    null_customers_removed = original_len - len(df)
    if null_customers_removed > 0:
        print(f"✅ Removed {null_customers_removed:,} rows with null Customer IDs")
    print(f"After removing null Customer IDs: {df.shape}")
    
    # Convert InvoiceDate to datetime
    print("📅 Converting InvoiceDate to datetime...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Remove rows with invalid dates
    original_len = len(df)
    df = df.dropna(subset=['InvoiceDate'])
    invalid_dates_removed = original_len - len(df)
    if invalid_dates_removed > 0:
        print(f"✅ Removed {invalid_dates_removed:,} rows with invalid dates")
    
    # Convert numeric columns
    print("🔢 Converting numeric columns...")
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Remove rows with invalid numeric values
    original_len = len(df)
    df = df.dropna(subset=['Quantity', 'Price'])
    invalid_numeric_removed = original_len - len(df)
    if invalid_numeric_removed > 0:
        print(f"✅ Removed {invalid_numeric_removed:,} rows with invalid numeric values")
    
    # Calculate total amount per transaction
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # Remove negative quantities and prices (keeping only positive transactions)
    original_len = len(df)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    negative_removed = original_len - len(df)
    if negative_removed > 0:
        print(f"✅ Removed {negative_removed:,} rows with negative quantities/prices")
    print(f"Final cleaned dataset shape: {df.shape}")
    
    # Basic data validation
    if len(df) == 0:
        raise ValueError("❌ No data remaining after preprocessing!")
    
    print(f"\n📊 DATA CLEANING SUMMARY:")
    total_removed = cancelled_removed + null_customers_removed + invalid_dates_removed + invalid_numeric_removed + negative_removed
    retention_rate = (len(df) / (len(df) + total_removed)) * 100
    
    print(f"  • Cancelled transactions removed: {cancelled_removed:,}")
    print(f"  • Null Customer IDs removed: {null_customers_removed:,}")
    print(f"  • Invalid dates removed: {invalid_dates_removed:,}")
    print(f"  • Invalid numeric values removed: {invalid_numeric_removed:,}")
    print(f"  • Negative values removed: {negative_removed:,}")
    print(f"  • Final dataset: {len(df):,} rows")
    print(f"  • Data retention rate: {retention_rate:.1f}%")
    
    print(f"\n📈 DATASET OVERVIEW:")
    print(f"  • Date range: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df['InvoiceDate'].max().strftime('%Y-%m-%d')}")
    print(f"  • Unique customers: {df['Customer ID'].nunique():,}")
    print(f"  • Unique invoices: {df['Invoice'].nunique():,}")
    print(f"  • Unique products: {df['StockCode'].nunique():,}")
    print(f"  • Countries: {df['Country'].nunique()}")
    print(f"  • Total transactions: {len(df):,}")
    print(f"  • Total revenue: ${df['TotalAmount'].sum():,.2f}")
    
    # Show sample of data
    print(f"\n🔍 SAMPLE DATA:")
    sample_data = df[['Invoice', 'Customer ID', 'Quantity', 'Price', 'TotalAmount', 'InvoiceDate']].head(3)
    print(sample_data.to_string())
    
    return df

def aggregate_customer_data(df):
    """
    Aggregate purchase data per customer using your exact column names
    """
    print("\n" + "=" * 60)
    print("STEP 2: CUSTOMER DATA AGGREGATION")
    print("=" * 60)
    
    if df is None or len(df) == 0:
        print("❌ No data to aggregate!")
        return None
    
    print("📊 Aggregating customer-level metrics...")
    
    # Get current date (max date in dataset)
    current_date = df['InvoiceDate'].max()
    print(f"Reference date for recency calculation: {current_date.strftime('%Y-%m-%d')}")
    
    try:
        # Customer-level aggregations using your column names
        print("🔄 Computing customer aggregations...")
        customer_data = df.groupby('Customer ID').agg({
            'TotalAmount': ['sum', 'mean', 'count'],  # Total spent, average order value, total transactions
            'Invoice': 'nunique',                     # Number of unique orders/invoices
            'InvoiceDate': ['min', 'max']            # First and last purchase dates
        }).reset_index()
        
        # Flatten column names
        customer_data.columns = [
            'CustomerID', 'TotalPurchaseAmount', 'AvgOrderValue', 
            'TotalTransactions', 'NumberOfOrders', 'FirstPurchaseDate', 'LastPurchaseDate'
        ]
        
        print(f"✅ Basic aggregations completed for {len(customer_data):,} customers")
        
        # Calculate derived metrics
        print("📅 Calculating recency (days since last purchase)...")
        customer_data['Recency'] = (current_date - customer_data['LastPurchaseDate']).dt.days
        
        print("⏱️  Calculating tenure (customer lifetime in days)...")
        customer_data['Tenure'] = (customer_data['LastPurchaseDate'] - customer_data['FirstPurchaseDate']).dt.days + 1
        
        print("🔄 Calculating frequency (purchase frequency)...")
        customer_data['Frequency'] = customer_data['NumberOfOrders'] / customer_data['Tenure']
        customer_data['Frequency'] = customer_data['Frequency'].fillna(0)
        
        # Handle edge cases and data quality issues
        print("🔧 Handling edge cases...")
        
        # Ensure non-negative values
        customer_data['Recency'] = customer_data['Recency'].clip(lower=0)
        customer_data['Tenure'] = customer_data['Tenure'].clip(lower=1)
        
        # Replace infinite values with NaN, then fill with appropriate defaults
        customer_data = customer_data.replace([np.inf, -np.inf], np.nan)
        customer_data['Frequency'] = customer_data['Frequency'].fillna(0)
        
        # Additional useful metrics
        print("📈 Calculating additional metrics...")
        customer_data['AvgDaysBetweenOrders'] = customer_data['Tenure'] / customer_data['NumberOfOrders'].clip(lower=1)
        customer_data['MonetaryValue'] = customer_data['TotalPurchaseAmount']  # Alias for RFM analysis
        
        # Drop date columns as they're no longer needed for modeling
        customer_data = customer_data.drop(['FirstPurchaseDate', 'LastPurchaseDate'], axis=1)
        
        print(f"✅ Customer-level dataset created: {customer_data.shape}")
        
        # Data quality check
        print(f"\n🔍 DATA QUALITY CHECK:")
        null_counts = customer_data.isnull().sum()
        if null_counts.sum() > 0:
            print("Null values found:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  • {col}: {count}")
        else:
            print("  ✅ No null values found!")
        
        print(f"  • Customers with zero purchase amount: {(customer_data['TotalPurchaseAmount'] <= 0).sum()}")
        print(f"  • Customers with zero orders: {(customer_data['NumberOfOrders'] <= 0).sum()}")
        print(f"  • Customers with negative recency: {(customer_data['Recency'] < 0).sum()}")
        
        print(f"\n📊 CUSTOMER METRICS SUMMARY:")
        print(f"  • Total customers: {len(customer_data):,}")
        print(f"  • Average total purchase: ${customer_data['TotalPurchaseAmount'].mean():,.2f}")
        print(f"  • Median total purchase: ${customer_data['TotalPurchaseAmount'].median():,.2f}")
        print(f"  • Average order value: ${customer_data['AvgOrderValue'].mean():.2f}")
        print(f"  • Average orders per customer: {customer_data['NumberOfOrders'].mean():.1f}")
        print(f"  • Average recency: {customer_data['Recency'].mean():.1f} days")
        print(f"  • Average tenure: {customer_data['Tenure'].mean():.1f} days")
        print(f"  • Average frequency: {customer_data['Frequency'].mean():.4f} orders/day")
        
        # Show distribution of key metrics
        print(f"\n📈 KEY METRICS DISTRIBUTION:")
        metrics_to_show = ['TotalPurchaseAmount', 'NumberOfOrders', 'Recency', 'Tenure']
        for metric in metrics_to_show:
            q25, q50, q75 = customer_data[metric].quantile([0.25, 0.5, 0.75])
            print(f"  • {metric}:")
            print(f"    - 25th percentile: {q25:.2f}")
            print(f"    - Median (50th): {q50:.2f}")
            print(f"    - 75th percentile: {q75:.2f}")
        
        print(f"\n🔍 SAMPLE OF PROCESSED CUSTOMERS:")
        sample_display = customer_data.head(5)[['CustomerID', 'TotalPurchaseAmount', 'AvgOrderValue', 'NumberOfOrders', 'Recency', 'Tenure']].round(2)
        print(sample_display.to_string())
        
        return customer_data
        
    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        print("📋 Debug info:")
        print(f"  • Customer ID data type: {df['Customer ID'].dtype}")
        print(f"  • Unique customers in raw data: {df['Customer ID'].nunique()}")
        print(f"  • Sample Customer IDs: {df['Customer ID'].unique()[:5]}")
        import traceback
        traceback.print_exc()
        return None

def save_processed_data(customer_data, filename='customer_data_preprocessed.csv'):
    """
    Save the preprocessed customer data with organized folder structure
    """
    if customer_data is None:
        print("❌ No data to save!")
        return False
    
    try:
        # Create organized folder structure
        folders_to_create = [
            'data',
            'data/processed',
            'models',
            'outputs',
            'outputs/plots'
        ]
        
        for folder in folders_to_create:
            os.makedirs(folder, exist_ok=True)
        
        # Define save locations
        save_locations = [
            os.path.join('data', 'processed', filename),
            filename  # Backup in current directory
        ]
        
        saved_files = []
        for filepath in save_locations:
            try:
                customer_data.to_csv(filepath, index=False)
                saved_files.append(filepath)
                print(f"✅ Data saved to: {filepath}")
            except Exception as e:
                print(f"⚠️  Could not save to {filepath}: {e}")
        
        if saved_files:
            print(f"\n📁 Successfully created {len(saved_files)} file(s)!")
            
            # Show file info
            main_file = saved_files[0]
            file_size = os.path.getsize(main_file)
            print(f"📊 File details:")
            print(f"  • Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"  • Rows: {len(customer_data):,}")
            print(f"  • Columns: {len(customer_data.columns)}")
            print(f"  • Column names: {', '.join(customer_data.columns)}")
            
            return True
        else:
            print(f"❌ Could not save data to any location!")
            return False
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

def main():
    """
    Main function to run data preprocessing for your specific dataset
    """
    print("🚀 CLV DATA PREPROCESSING - CUSTOM COLUMN VERSION")
    print("=" * 80)
    print("Dataset columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country")
    print("=" * 80)
    
    try:
        # Step 1: Load and preprocess data
        file_path = 'online_retail.csv'
        
        if not os.path.exists(file_path):
            print(f"❌ File '{file_path}' not found in current directory!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return
        
        df = load_and_preprocess_data(file_path)
        
        if df is None:
            print("❌ Failed to load and preprocess data. Exiting...")
            return
        
        # Step 2: Aggregate customer data
        customer_data = aggregate_customer_data(df)
        
        if customer_data is None:
            print("❌ Failed to aggregate customer data. Exiting...")
            return
        
        # Step 3: Save processed data
        success = save_processed_data(customer_data)
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            print(f"\n📋 PROCESSING SUMMARY:")
            print(f"  ✅ Input file: {file_path}")
            print(f"  ✅ Customers processed: {len(customer_data):,}")
            print(f"  ✅ Features created: {len(customer_data.columns)}")
            print(f"  ✅ Output file: customer_data_preprocessed.csv")
            
            print(f"\n📊 DATASET READY FOR:")
            print("  • Feature selection")
            print("  • Customer Lifetime Value (CLV) modeling")
            print("  • Customer segmentation")
            print("  • Predictive analytics")
            
            print(f"\n🚀 NEXT STEPS:")
            print("1. Run: python 2_feature_selection.py")
            print("2. Or explore the data: head customer_data_preprocessed.csv")
            print("3. Check the data/processed/ folder for organized files")
            
            # Display final comprehensive statistics
            print(f"\n📊 FINAL CUSTOMER DATASET STATISTICS:")
            print("=" * 60)
            
            stats = customer_data.describe()
            
            # Show key statistics in a more readable format
            key_metrics = ['TotalPurchaseAmount', 'AvgOrderValue', 'NumberOfOrders', 'Recency', 'Tenure', 'Frequency']
            
            for metric in key_metrics:
                if metric in stats.columns:
                    print(f"\n{metric}:")
                    print(f"  Mean: {stats.loc['mean', metric]:.2f}")
                    print(f"  Median: {stats.loc['50%', metric]:.2f}")
                    print(f"  Std: {stats.loc['std', metric]:.2f}")
                    print(f"  Min: {stats.loc['min', metric]:.2f}")
                    print(f"  Max: {stats.loc['max', metric]:.2f}")
            
        else:
            print("❌ Failed to save processed data.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user.")
        
    except Exception as e:
        print(f"\n💥 PREPROCESSING FAILED: {str(e)}")
        print(f"\n🔍 ERROR DETAILS:")
        import traceback
        traceback.print_exc()
        
        print(f"\n🛠️  TROUBLESHOOTING CHECKLIST:")
        print("1. ✅ File 'online_retail.csv' exists in current directory")
        print("2. ✅ File has columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country")
        print("3. ✅ You have read/write permissions")
        print("4. ✅ Sufficient memory available")
        print("5. ✅ All required Python packages installed")

if __name__ == "__main__":
    main()