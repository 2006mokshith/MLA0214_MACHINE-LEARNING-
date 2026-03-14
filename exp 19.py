# Bank Loan Prediction using Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Bank Loan dataset
data = {
    'Income':[25000,40000,50000,60000,80000,120000,30000,70000,90000,45000],
    'Age':[25,35,45,50,23,40,28,42,36,31],
    'Loan':[0,0,1,1,0,1,0,1,1,0]  # 0 = No Loan, 1 = Loan Approved
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Income','Age']]
y = df['Loan']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predicted Loan Approval:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
