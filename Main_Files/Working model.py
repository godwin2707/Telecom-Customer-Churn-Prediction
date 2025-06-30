import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer # to proceed with OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset=pd.read_csv("D:\Projects\Churn Prediction\Dataset 1\Telco Customer Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Converting TotalCharges to numeric
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

# Droping rows with missing TotalCharges values
dataset.dropna(subset=["TotalCharges"], inplace=True)

# Droping the customerID column
if 'customerID' in dataset.columns:
    dataset.drop("customerID", axis=1, inplace=True)
#assigning the features in the X and the labels in y
X = dataset.drop("Churn", axis=1)
y = dataset["Churn"]

# Encoding the target (Churn)
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Identifyingcategorical and numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(exclude=['object']).columns.tolist()

# Label encoding all categorical features for compatibility
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ColumnTransformer ]
column_transformer = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    ('num', 'passthrough', numeric_columns)
])

# Fiting and transforming  the training data
X_train_transformed = column_transformer.fit_transform(X_train)

# Training the model 
model = RandomForestClassifier(random_state=42)
model.fit(X_train_transformed, y_train)

# Loading the model
with open("rf.pkl", "wb") as f:
    pickle.dump(model, f)

with open("column_encoder.pkl", "wb") as f:
    pickle.dump(column_transformer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

