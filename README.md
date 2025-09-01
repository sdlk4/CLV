# Customer Lifetime Value (CLV) Prediction  
Customer Lifetime Value prediction is crucial for any business that wants to:
- Focus marketing efforts on high-value customers
- Optimize resource allocation
- Make data-driven decisions about customer acquisition costs
- Build targeted retention strategies
This project uses machine learning to predict CLV and then segments customers into high, medium, and low value groups. No more guesswork – just solid predictions backed by data.

## Project Structure  
- ├── data/processed/          # Clean, processed datasets ready for modeling
- ├── models/                  # Trained model files
- ├── outputs/                 # Generated predictions and visualizations
- ├── results/                 # Analysis results and performance metrics
- ├── 1_data_processing.py     # Data cleaning and preprocessing
- ├── 2_feature_selection.py   # Feature engineering and selection
- ├── 3_model_training.py      # Model training and validation
- ├── 4_model_saving_prediction.py  # Save models and generate predictions
- ├── 5_customer_segmentation.py    # Customer segmentation logic
- ├── 6_automation_script.py   # End-to-end pipeline automation
- └── requirements.txt         # Project dependencies

## Getting Started
## Prerequisites
You'll need Python 3.7+ and the packages listed in requirements.txt. The usual suspects are there:
- pandas, numpy for data manipulation
- scikit-learn for machine learning
- xgboost for gradient boosting
- matplotlib, seaborn for visualization

# Installation
1. Clone this repository:
- git clone https://github.com/sdlk4/Customer-Lifetime-Value-Prediction.git
- cd Customer-Lifetime-Value-Prediction

2. Install the required packages:
- pip install -r requirements.txt

3. Run the automated pipeline:
- python 6_automation_script.py
That's it! The script will handle everything from data processing to final predictions.

# How It Works
1. Data Processing (1_data_processing.py)
- Loads and cleans raw customer data
- Handles missing values and outliers
- Creates derived features like recency, frequency, and monetary values

2. Feature Selection (2_feature_selection.py)
- Identifies the most predictive features for CLV
- Removes redundant or low-impact variables
- Prepares the final feature set for modeling

3. Model Training (3_model_training.py)
- Trains multiple regression models (Linear, Random Forest, XGBoost)
- Performs cross-validation and hyperparameter tuning
- Selects the best-performing model based on evaluation metrics

4. Predictions & Model Saving (4_model_saving_prediction.py)
- Saves the trained model for future use
- Generates CLV predictions for all customers
- Creates prediction confidence intervals

5. Customer Segmentation (5_customer_segmentation.py)
- Segments customers into High, Medium, and Low value groups
- Generates segment profiles and characteristics
- Creates actionable insights for each segment

6. Automation (6_automation_script.py)
- Runs the entire pipeline with a single command
- Handles error checking and logging
- Perfect for scheduled runs or production deployment

# Key Features
- Automated Feature Engineering: Automatically creates RFM (Recency, Frequency, Monetary) features and other relevant predictors
- Multiple Model Comparison: Tests various algorithms to find the best performer for your data
- Customer Segmentation: Goes beyond predictions to provide actionable customer segments
- Production Ready: Includes model saving/loading and batch prediction capabilities
- Comprehensive Evaluation: Multiple metrics and visualizations to assess model performance

# Sample Output
The project generates several key outputs:
- clv_predictions.csv: Customer-level CLV predictions
- customer_segments.csv: Customer segmentation results
- model_performance_comparison.csv: Model evaluation metrics
- Various visualization plots showing model performance and segment characteristics

# Model Performance
The models are evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Cross-validation scores
Typical performance on retail datasets shows R² scores above 0.75, but your mileage may vary depending on data quality and business context.

# Contributing
Found a bug or have an idea for improvement? Feel free to:
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request
  
# Technical Notes
- The project uses XGBoost as the default algorithm due to its strong performance on tabular data
- Feature scaling is handled automatically within the pipeline
- The segmentation uses quantile-based thresholds but can be customized for business needs

- Predicted CLV: $1,245.67
