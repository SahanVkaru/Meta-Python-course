from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

print(df.shape)

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1) # features (input)
y = df['target'] # target variable (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() # create an instance of the LogisticRegression class
model.fit(X_train, y_train) # fit the model to the training data

# Make predictions and evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test) # make predictions on the test set

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))