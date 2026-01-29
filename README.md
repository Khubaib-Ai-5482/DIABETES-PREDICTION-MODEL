#Diabetes Prediction using Random Forest

This project builds a machine learning model to predict diabetes using a healthcare dataset. The model is trained with a Random Forest Classifier, and exploratory data analysis (EDA) is performed to understand the dataset and feature relationships.

##Dataset

The dataset contains patient information such as age, gender, and other health metrics, with the target variable diabetes (0 = No, 1 = Yes). Categorical features are encoded for model training.

###Features
age
gender
Other health-related features (depending on dataset columns)
diabetes (target variable)

###Project Workflow

Data Loading: The dataset is imported using Pandas.

Preprocessing:

Categorical variables are encoded using LabelEncoder.

Features and target variable are separated.

Train-test split with stratification to preserve class distribution.

Model Training: Random Forest Classifier is trained with 100 trees and a max depth of 10.

Evaluation:

Model accuracy and classification report are generated.

Confusion matrix is visualized.

Exploratory Data Analysis (EDA):

Diabetes distribution by gender

Age vs diabetes scatter plot

Trend of diabetes labels over dataset index

Feature correlation heatmap

Feature importance from Random Forest

Results

Accuracy: [Insert your model accuracy here]

Confusion matrix and classification report are generated to evaluate model performance.

Feature importance highlights the most predictive features for diabetes.

Visualizations

Confusion matrix heatmap

Diabetes distribution by gender

Age vs diabetes scatter plot

Diabetes label trend plot

Feature correlation heatmap

Feature importance bar chart

Usage

Clone the repository:

git clone https://github.com/your-username/diabetes-prediction.git

Install dependencies:

pip install pandas matplotlib seaborn scikit-learn

Run the Python script:

python diabetes_prediction.py
Insights

The model can identify patients at risk of diabetes with reasonable accuracy.

Feature importance provides insights into which factors contribute most to diabetes prediction.

The visualizations help in understanding the data distribution and correlations.

If you want, I can also write a shorter, more visually appealing version with badges, dataset link, and GIF plots—which looks very professional on GitHub. This will make your repo look like a real data scientist’s portfolio.

Do you want me to do that too?
