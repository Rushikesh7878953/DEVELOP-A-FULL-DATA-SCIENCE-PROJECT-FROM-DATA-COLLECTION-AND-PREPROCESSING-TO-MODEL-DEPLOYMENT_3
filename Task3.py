# Load Iris Data
from sklearn.datasets import load_iris
import pandas as pd

iris=load_iris()

# Convering Iris Data Into A DataFrame
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target

print(df.head())

from sklearn.preprocessing import StandardScaler

#Use For Check The Missing Values
print(df.isnull().sum())

# Standardize The Features
scaler=StandardScaler()
X=scaler.fit_transform(df.drop('target',axis=1))
y=df['target']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Spliting The Data Into Train and Test Sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Train The Model
model=LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate The Model
print(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

import joblib

# Use To Save The Trained Model and Scaler
joblib.dump(model,'iris_model.pkl')
joblib.dump(scaler,'scaler.pkl')

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Loading The Pre-Trained Model And Scaler
model=joblib.load('iris_model.pkl')
scaler=joblib.load('scaler.pkl')

# FastAPI App Instance
app=FastAPI()

# Define The Input Data Structure
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input_data: IrisInput):
    # Prepare The Input Data For Prediction
    features = np.array([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]])

    # Scale The Features
    features_scaled = scaler.transform(features)

    # Predict Using The Trained Model
    prediction = model.predict(features_scaled)

    # Map The Prediction To The Species Name
    species = ['setosa', 'versicolor', 'virginica']
    predicted_species=species[prediction[0]]

    return {"species": predicted_species}

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Flower Species Prediction API!"}

curl_command = """curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [5.1, 3.5, 1.4, 0.2]
}'"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load The Pre-Trained Model And Scaler
model=joblib.load('iris_model.pkl')
scaler=joblib.load('scaler.pkl')

app=Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Iris Flower Species Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])

    # Scale The Features
    features_scaled = scaler.transform(features)

    # Make Prediction
    prediction = model.predict(features_scaled)

    # Map The Prediction To Species
    species = ['setosa', 'versicolor', 'virginica']
    predicted_species = species[prediction[0]]

    return jsonify({'species': predicted_species})

if __name__ == '__main__':
    app.run(debug=True)