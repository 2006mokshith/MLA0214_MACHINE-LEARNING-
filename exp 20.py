# Future Sales Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sales dataset
data = {
    'Month':[1,2,3,4,5,6,7,8,9,10],
    'Sales':[100,120,130,150,170,180,200,210,230,250]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict future sales
future_month = [[11]]
predicted_sales = model.predict(future_month)

print("Predicted Sales for Month 11:", predicted_sales[0])
