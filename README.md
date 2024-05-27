# Breast Cancer Prediction Analysis
This repository contains a Jupyter notebook for analyzing a breast cancer dataset and predicting whether a tumor is malignant or benign using machine learning techniques.

## Installation and Requirements
To run the notebook, you need to have Python and Jupyter Notebook installed. The required libraries are listed below:

<ul>
<li>pandas</li>
<li>numpy</li>
<li>matplotlib</li>
<li>seaborn</li>
<li>scikit-learn</li>
<li>xgboost</li>
</ul>
You can install the required packages using pip:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Usage
To use the notebook, simply clone this repository and open the Jupyter notebook:

git clone <https://github.com/EstherNjuguna/breast_cancer_prediction><br>
cd <breast_cancer_prediction><br>
jupyter notebook breast.ipynb<br>
## Dataset Information
The dataset used in this analysis is the Breast Cancer Wisconsin dataset. It contains 683 instances with 11 attributes. The attributes include:

- Sample code number: id number
- Clump Thickness: 1 - 10
- Uniformity of Cell Size: 1 - 10
- Uniformity of Cell Shape: 1 - 10
- Marginal Adhesion: 1 - 10
- Single Epithelial Cell Size: 1 - 10
- Bare Nuclei: 1 - 10
- Bland Chromatin: 1 - 10
- Normal Nucleoli: 1 - 10
- Mitoses: 1 - 10
- Class: (2 for benign, 4 for malignant)
## Analysis Steps
The notebook includes the following steps:

- Data Loading and Inspection: Load the dataset and display basic information.
- Data Cleaning: Handle missing values and convert data types.
- Exploratory Data Analysis (EDA): Visualize the data to understand distributions and relationships.
- Feature Selection: Identify important features for the prediction model.
- Model Building: Build and evaluate machine learning models (e.g., Logistic Regression, XGBoost, Random Forest, Linear Regression).
- Model Evaluation: Evaluate the performance of the models using metrics like accuracy, precision, recall, and F1-score.
## Results
In this analysis, multiple machine learning models were trained and evaluated to predict whether a breast tumor is malignant or benign. The models include:

- Logistic Regression
- XGBoost (Gradient Boosting)
- Random Forest
- Linear Regression
###  Logistic Regression
Accuracy: 95.6 <br>
<b>Precision:</b> <br>
- Benign (Class 2): 96%
- Malignant (Class 4): 96%<br>
<b>Recall:</b> <br>
- Benign (Class 2): 98%
- Malignant (Class 4): 92%<br>
<b>F1-Score:</b> <br>
- Benign (Class 2): 97%
- Malignant (Class 4): 94%
### XGBoost (Gradient Boosting)
<b>Best Parameters:</b><br>
- colsample_bytree: 0.6
- learning_rate: 0.2
- max_depth: 3
- min_child_weight: 3
- n_estimators: 300
- subsample: 0.6<br>
<b>Cross-Validation Accuracy:</b> 97.6%<br>
<b> Test Set Accuracy:</b> 97.1%<br>
### Random Forest
<b>- Accuracy:</b> 96.4% <br>
<b> - Precision:</b> <br>
- Benign (Class 2): 97%
- Malignant (Class 4): 96%<br>
<b>Recall:</b> <br>
- Benign (Class 2): 98%
- Malignant (Class 4): 93%<br>
<b>F1-Score:</b> <br>
- Benign (Class 2): 97.5%
- Malignant (Class 4): 94.5%
### Linear Regression
<b>Accuracy:</b> 94.8% <br>
<b>Precision:</b>
- Benign (Class 2): 95%
 - Malignant (Class 4): 95%<br>
<b> Recall:</b> <br>
- Benign (Class 2): 97%
- Malignant (Class 4): 92%<br>
<b> F1-Score: </b> <br>
- Benign (Class 2): 96%
- Malignant (Class 4): 93.5%
## Best Model
The XGBoost (Gradient Boosting) model achieved the highest test set accuracy of 97.1%, making it the best-performing model for this dataset.

## Visualizations
The notebook includes various visualizations to aid in understanding the data and the performance of the models. Some key visualizations are:

- Confusion Matrix: Shows the performance of the classification model by illustrating the true positive, true negative, false positive, and false negative predictions.
- ROC Curve: Plots the true positive rate against the false positive rate to evaluate the trade-off between sensitivity and specificity.
- Feature Importance: Displays the importance of each feature in the XGBoost model, helping to understand which features contribute most to the predictions.


Conclusion
The analysis demonstrates that machine learning models, particularly the XGBoost model, can effectively predict the malignancy of breast tumors with high accuracy. The XGBoost model showed the best performance, making it a suitable choice for this classification task. However, it is essential to consider the specific requirements and constraints of the application domain when choosing a model.
