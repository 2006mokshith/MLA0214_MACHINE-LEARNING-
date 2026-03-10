# Car Price Prediction using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Sample dataset
data = {
    'Year': [2014,2013,2017,2011,2014,2012,2015,2016],
    'Present_Price': [5.59,9.54,9.85,4.15,6.87,3.51,5.00,7.45],
    'Kms_Driven': [27000,43000,6900,5200,42450,35000,40000,15000],
    'Fuel_Type': [0,1,1,1,1,0,1,1],      # 0=Petrol,1=Diesel
    'Seller_Type': [0,0,0,0,0,0,0,0],    # 0=Dealer
    'Transmission': [0,0,0,0,0,1,0,0],   # 0=Manual,1=Automatic
    'Selling_Price': [3.35,4.75,7.25,2.85,4.60,2.25,3.90,5.95]
}

df = pd.DataFrame(data)

# Input and Output
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Actual Price:", y_test.values)
print("Predicted Price:", y_pred)

# Accuracy
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
