from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#sample dataset
data = pd.DataFrame({
    'Score1': [85, 60, 90, 45],
    'Score2': [80, 85, 92, 50],
    'admitted': [1, 0, 1, 0]
})

X = data[['Score1', 'Score2']] # features (input)
y = data['admitted'] # target variable (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression() # create an instance of the LinearRegression class

model.fit(X_train, y_train) # fit the model to the training data

predictions = model.predict(X_test) # make predictions on the test set

print("Accuracy:", accuracy_score(y_test, predictions)) # round the predictions to get binary output
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions)) # round the predictions to get binary output