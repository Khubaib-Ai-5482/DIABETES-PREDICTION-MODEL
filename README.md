# Diabetes Prediction Project 🩺

Predict diabetes using patient data with a **Random Forest Classifier** and explore important features using data visualization.  

---

## Overview

This project predicts whether a patient has diabetes based on various health metrics. It includes:  

- **Data preprocessing**: Label encoding for categorical features.  
- **Exploratory Data Analysis (EDA)**: Histograms, scatter plots, correlation heatmaps.  
- **Modeling**: Random Forest Classifier.  
- **Evaluation**: Accuracy, classification report, confusion matrix.  
- **Feature Importance**: Understand which features contribute most to predictions.  

---

## Dataset

The dataset contains health-related features such as:  

- `age`  
- `gender`  
- Various medical measurements  
- `diabetes` (target variable: 0 = No, 1 = Yes)  

> Place `diabetes_prediction_dataset.csv` in the project folder before running the code.

---

## Usage

1. Clone the repository:  
```bash
git clone https://github.com/your-username/diabetes-prediction.git
```  

2. Install dependencies:  
```bash
pip install pandas matplotlib seaborn scikit-learn
```  

3. Run the Python script:  
```bash
python diabetes_prediction.py
```  

---

## Results & Visualizations

- **Model Accuracy**: ~[insert your accuracy]%  
- **Confusion Matrix**  
- **Diabetes Distribution by Gender**  
- **Age vs Diabetes Scatter Plot**  
- **Trend of Diabetes Labels**  
- **Feature Correlation Heatmap**  
- **Feature Importance Plot**  

> All plots are generated automatically by the script.  

---

## Insights

- Random Forest identifies high-risk patients effectively.  
- Age, gender, and other medical features strongly influence diabetes prediction.  
- Visualizations help understand patterns and correlations in the data.  
