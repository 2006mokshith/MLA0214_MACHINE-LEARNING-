# Mobile Price Prediction using Decision Tree Algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Mobile dataset
data = {
    'RAM':[2,3,4,6,8,12,4,6,8,3],
    'Storage':[16,32,64,64,128,256,128,64,256,32],
    'Battery':[3000,3200,3500,4000,4500,5000,4200,3800,4700,3100],
    'Price':[0,0,1,1,2,2,2,1,2,0]   # 0=Low, 1=Medium, 2=High
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['RAM','Storage','Battery']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Test Data:")
print(X_test)

print("\nPredicted Price Category:")
print(y_pred)

print("\nActual Price Category:")
print(y_test.values)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
