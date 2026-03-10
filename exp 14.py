import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# House dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4],
    "Price": [200000, 250000, 300000, 320000, 360000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)

# Input features
X = df[['Area','Bedrooms','Bathrooms']]

# Target
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
predicted_price = model.predict(X_test)

print("Actual Price:", list(y_test))
print("Predicted Price:", predicted_price)

# Error
error = mean_absolute_error(y_test, predicted_price)
print("Mean Absolute Error:", error)
