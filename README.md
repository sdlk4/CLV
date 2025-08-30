# 🧮 Customer Lifetime Value (CLV) Prediction  

## 📌 Project Overview  
This project builds a **Customer Lifetime Value (CLV) Prediction Model** using regression techniques in Python.  
It helps businesses **identify high-value customers** by analyzing purchase history, demographics, and behavioral data.  
The pipeline automates:
- Data preprocessing  
- Feature engineering & selection  
- Model training & evaluation  
- Customer segmentation  
- End-to-end automation  

## 📂 Project Structure  
CLV_PREDICTION/
│── data/                           # Raw and processed datasets
│── models/                         # Saved ML models
│── outputs/                        # Generated reports, plots, results
│   ├── plots/                      # Visualizations (feature importance, comparison, etc.)
│   ├── results/                    # Prediction results & analysis outputs
│── 1_data_processing.py            # Data cleaning & preprocessing
│── 2_feature_selection.py          # Feature engineering & selection
│── 3_model_training.py             # Train ML models (Linear Regression, XGBoost, etc.)
│── 4_model_saving_prediction.py    # Save model & make predictions
│── 5_customer_segmentation.py      # Customer segmentation based on CLV
│── 6_automation_script.py          # End-to-end pipeline automation
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation

## 📊 Dataset
You can use any **customer transaction dataset** that includes:  
- `CustomerID`  
- `TransactionDate`  
- `PurchaseAmount`  

Example: [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)  

## 📈 Exploratory Data Analysis
### Customer Spend Distribution
- Most customers have **low average spend**, with a few **high-value customers**.  

### Model Performance
- **Random Forest & XGBoost outperform Linear Regression**.  
- Random Forest achieves the **lowest RMSE and highest R² score**.  

## 🛠️ Installation
git clone https://github.com/sdlk/CLV.git
cd CLV
Install dependencies:
pip install -r requirements.txt

## Generate visualizations and logs
Run scripts individually:
- python 1_data_analysis.py
- python 2_feature_engineering.py
- python 3_model_training.py
- python 4_model_prediction.py

## 📊 Results
- Random Forest R²: ~0.92
- XGBoost R²: ~0.90
- Linear Regression R²: ~0.75

## Metrics: RMSE, MAE, R² comparison across models.
Feature Importance shows Recency, Frequency, and Monetary Value are strong predictors of CLV.

## Example Prediction Output:
- Customer ID: 205
- Predicted CLV: $1,245.67
