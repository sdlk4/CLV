import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_predictions(file_path='clv_predictions.csv'):
    """
    Load CLV predictions from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded predictions data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run prediction script first.")
        return None

def create_clv_segments_percentile(df, clv_column='Predicted_CLV'):
    """
    Segment customers into tiers based on predicted CLV using percentiles
    - High Value (top 20%)
    - Medium Value (middle 50%) 
    - Low Value (bottom 30%)
    """
    print("\nCreating CLV segments based on percentiles...")
    
    # Calculate percentile thresholds
    high_threshold = df[clv_column].quantile(0.8)  # Top 20%
    low_threshold = df[clv_column].quantile(0.3)   # Bottom 30%
    
    print(f"High Value threshold (80th percentile): ${high_threshold:.2f}")
    print(f"Low Value threshold (30th percentile): ${low_threshold:.2f}")
    
    # Create segments
    def assign_segment(clv_value):
        if clv_value >= high_threshold:
            return 'High Value'
        elif clv_value >= low_threshold:
            return 'Medium Value'
        else:
            return 'Low Value'
    
    df['CLV_Segment'] = df[clv_column].apply(assign_segment)
    
    # Display segment distribution
    segment_counts = df['CLV_Segment'].value_counts()
    segment_percentages = df['CLV_Segment'].value_counts(normalize=True) * 100
    
    print("\n=== Segment Distribution ===")
    for segment in ['High Value', 'Medium Value', 'Low Value']:
        count = segment_counts.get(segment, 0)
        percentage = segment_percentages.get(segment, 0)
        print(f"{segment}: {count} customers ({percentage:.1f}%)")
    
    return df

def create_clv_segments_kmeans(df, clv_column='Predicted_CLV', n_clusters=3):
    """
    Alternative segmentation using K-Means clustering
    """
    print(f"\nCreating CLV segments using K-Means (k={n_clusters})...")
    
    # Prepare data for clustering
    X = df[[clv_column]]
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to dataframe
    df['CLV_Cluster'] = cluster_labels
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()
    cluster_info = []
    
    for i in range(n_clusters):
        cluster_mask = df['CLV_Cluster'] == i
        cluster_data = df[cluster_mask]
        
        cluster_info.append({
            'Cluster': i,
            'Count': len(cluster_data),
            'Avg_CLV': cluster_data[clv_column].mean(),
            'Min_CLV': cluster_data[clv_column].min(),
            'Max_CLV': cluster_data[clv_column].max(),
            'Center': cluster_centers[i]
        })
    
    # Sort by average CLV and assign meaningful names
    cluster_info_df = pd.DataFrame(cluster_info).sort_values('Avg_CLV', ascending=False)
    
    # Map clusters to meaningful names
    cluster_mapping = {}
    segment_names = ['High Value', 'Medium Value', 'Low Value']
    
    for idx, row in cluster_info_df.iterrows():
        if idx < len(segment_names):
            cluster_mapping[row['Cluster']] = segment_names[idx]
        else:
            cluster_mapping[row['Cluster']] = f'Segment_{idx+1}'
    
    df['CLV_Segment_KMeans'] = df['CLV_Cluster'].map(cluster_mapping)
    
    print("\n=== K-Means Cluster Information ===")
    print(cluster_info_df.round(2))
    
    return df, cluster_info_df

def analyze_segments(df, clv_column='Predicted_CLV'):
    """
    Analyze characteristics of each segment
    """
    print("\n=== Segment Analysis ===")
    
    # Segment statistics
    segment_stats = df.groupby('CLV_Segment')[clv_column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    print("\nSegment Statistics:")
    print(segment_stats)
    
    # Calculate total CLV by segment
    total_clv_by_segment = df.groupby('CLV_Segment')[clv_column].sum()
    total_clv = df[clv_column].sum()
    
    print(f"\nTotal CLV Distribution:")
    for segment, clv_sum in total_clv_by_segment.items():
        percentage = (clv_sum / total_clv) * 100
        print(f"{segment}: ${clv_sum:,.2f} ({percentage:.1f}% of total CLV)")
    
    return segment_stats

def visualize_segments(df, clv_column='Predicted_CLV'):
    """
    Create visualizations for customer segments
    """
    print("\nCreating segment visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of CLV by segment (box plot)
    sns.boxplot(data=df, x='CLV_Segment', y=clv_column, ax=axes[0,0])
    axes[0,0].set_title('CLV Distribution by Segment')
    axes[0,0].set_ylabel('Predicted CLV ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Count of customers by segment
    segment_counts = df['CLV_Segment'].value_counts()
    axes[0,1].bar(segment_counts.index, segment_counts.values)
    axes[0,1].set_title('Number of Customers by Segment')
    axes[0,1].set_ylabel('Number of Customers')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. CLV histogram with segment colors
    for segment in df['CLV_Segment'].unique():
        segment_data = df[df['CLV_Segment'] == segment][clv_column]
        axes[1,0].hist(segment_data, alpha=0.7, label=segment, bins=30)
    
    axes[1,0].set_xlabel('Predicted CLV ($)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('CLV Distribution by Segment')
    axes[1,0].legend()
    
    # 4. Pie chart of segment distribution
    segment_counts = df['CLV_Segment'].value_counts()
    axes[1,1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Customer Segment Distribution')
    
    plt.tight_layout()
    plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_segment_profiles(df, clv_column='Predicted_CLV'):
    """
    Create detailed profiles for each segment
    """
    print("\n=== Creating Segment Profiles ===")
    
    profiles = {}
    
    for segment in df['CLV_Segment'].unique():
        segment_data = df[df['CLV_Segment'] == segment]
        
        profile = {
            'Segment': segment,
            'Customer_Count': len(segment_data),
            'Percentage_of_Total': (len(segment_data) / len(df)) * 100,
            'Avg_CLV': segment_data[clv_column].mean(),
            'Median_CLV': segment_data[clv_column].median(),
            'Min_CLV': segment_data[clv_column].min(),
            'Max_CLV': segment_data[clv_column].max(),
            'Total_CLV': segment_data[clv_column].sum(),
            'CLV_Contribution': (segment_data[clv_column].sum() / df[clv_column].sum()) * 100
        }
        
        profiles[segment] = profile
    
    # Convert to DataFrame for better display
    profiles_df = pd.DataFrame(profiles).T
    
    print(profiles_df.round(2))
    
    return profiles_df

def recommend_strategies(profiles_df):
    """
    Recommend marketing strategies for each segment
    """
    print("\n=== Marketing Strategy Recommendations ===")
    
    strategies = {
        'High Value': {
            'Description': 'Top 20% customers with highest CLV',
            'Strategies': [
                'VIP customer service and dedicated account management',
                'Exclusive product launches and early access',
                'Premium loyalty rewards and personalized offers',
                'Regular engagement to prevent churn',
                'Upselling premium products and services'
            ]
        },
        'Medium Value': {
            'Description': 'Middle 50% customers with moderate CLV',
            'Strategies': [
                'Targeted campaigns to increase purchase frequency',
                'Cross-selling complementary products',
                'Loyalty programs to encourage repeat purchases',
                'Email marketing with personalized recommendations',
                'Seasonal promotions and discounts'
            ]
        },
        'Low Value': {
            'Description': 'Bottom 30% customers with lowest CLV',
            'Strategies': [
                'Cost-effective digital marketing campaigns',
                'Basic loyalty programs and incentives',
                'Focus on customer education and engagement',
                'Win-back campaigns for inactive customers',
                'Automated marketing with minimal manual intervention'
            ]
        }
    }
    
    for segment, info in strategies.items():
        if segment in profiles_df.index:
            print(f"\n{segment} Customers:")
            print(f"Description: {info['Description']}")
            print(f"Count: {profiles_df.loc[segment, 'Customer_Count']:.0f} customers")
            print(f"Avg CLV: ${profiles_df.loc[segment, 'Avg_CLV']:.2f}")
            print("Recommended Strategies:")
            for strategy in info['Strategies']:
                print(f"  • {strategy}")

def save_segmented_data(df, output_file='customer_clv_segments.csv'):
    """
    Save the segmented customer data
    """
    # Select relevant columns
    columns_to_save = ['CustomerID', 'Predicted_CLV', 'CLV_Segment']
    
    # Add KMeans segment if it exists
    if 'CLV_Segment_KMeans' in df.columns:
        columns_to_save.append('CLV_Segment_KMeans')
    
    output_df = df[columns_to_save]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nSegmented data saved to '{output_file}'")
    
    return output_df

def main():
    """
    Main function to perform customer segmentation
    """
    # Load CLV predictions
    df = load_predictions()
    
    if df is None:
        return
    
    # Create segments using percentile method
    df = create_clv_segments_percentile(df)
    
    # Alternative: Create segments using K-Means
    df, cluster_info = create_clv_segments_kmeans(df)
    
    # Analyze segments
    segment_stats = analyze_segments(df)
    
    # Create visualizations
    visualize_segments(df)
    
    # Create detailed profiles
    profiles_df = create_segment_profiles(df)
    
    # Recommend strategies
    recommend_strategies(profiles_df)
    
    # Save segmented data
    output_df = save_segmented_data(df)
    
    # Save analysis results
    segment_stats.to_csv('segment_analysis_stats.csv')
    profiles_df.to_csv('segment_profiles.csv')
    
    print("\n=== Segmentation Complete ===")
    print("Files created:")
    print("- customer_clv_segments.csv (segmented customer data)")
    print("- segment_analysis_stats.csv (segment statistics)")
    print("- segment_profiles.csv (detailed segment profiles)")
    print("- customer_segmentation_analysis.png (visualizations)")
    
    return df, profiles_df

if __name__ == "__main__":
    df, profiles = main()