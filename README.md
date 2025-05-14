The Telecom Customer Churn Prediction project is a machine learning application designed to predict whether a telecom customer is likely to churn based on their account information and usage behavior. The goal is to help telecom businesses proactively identify at-risk customers and take necessary actions to improve retention.

🔍 Project Overview
This web-based app allows users to input customer details and instantly receive a churn prediction, along with the probability score. The app is interactive, built using Streamlit, and also saves prediction results to a MySQL database for tracking and analysis.

❓ What problem does it solve?
It helps telecom providers reduce customer attrition by offering real-time churn risk insights, which improves decision-making around customer support, marketing, and loyalty programs. It significantly cuts down manual analysis and allows businesses to act before a customer leaves.

🧠 Key Components
Frontend/UI
Built using Streamlit, a Python framework for interactive web apps.
Users can enter customer attributes (gender, tenure, internet service, etc.) via dropdowns and number inputs.
Displays prediction result and churn probability clearly to the user.

Model & Preprocessing
Random Forest Classifier trained using scikit-learn for robust prediction.
Categorical variables are preprocessed using LabelEncoder and OneHotEncoder.
Data is vectorized using a ColumnTransformer to handle both categorical and numeric features.
Predictions and probabilities are generated and interpreted directly in the app.

Prediction Logging
Integrates with MySQL to store input data and model predictions.
Automatically creates a table (if not present) and logs each prediction with full customer detail for future analysis.

🛠 Tech Stack
Python
Streamlit – For the interactive web interface
scikit-learn – For training the machine learning model
pandas – For data manipulation
pickle – For saving and loading models and encoders
MySQL – For storing prediction logs
LabelEncoder & OneHotEncoder – For preprocessing categorical data
