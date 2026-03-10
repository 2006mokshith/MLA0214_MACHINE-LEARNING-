# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'Income': [25000, 30000, 40000, 50000, 60000, 70000, 80000],
    'Loan': [10000, 12000, 15000, 20000, 25000, 30000, 35000],
    'CreditScore': ['Low', 'Low', 'Medium', 'Medium', 'High', 'High', 'High']
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Income', 'Loan']]
y = df['CreditScore']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Predicted Credit Score:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))