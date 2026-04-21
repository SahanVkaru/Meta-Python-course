import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the dataset

#Simulate a dataset
np.random.seed(0) # get the same random numbers every time you run the code
X1 = np.random.rand(100) * 10 # feature 1 (input)
X2 = np.random.rand(100) * 5 # feature 2 (input)
noise = np.random.rand(100) # random noise to make the data more realistic
Y = 3 + 2 * X1 + 4 * X2 + noise # target variable (output)

#Create DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
df.head() # print the first 5 rows of the dataframe

# 3. Explortory Data Analysis (EDA)

#summary
print(df.describe()) 

#visualization
sns.pairplot(df)
plt.show()

#Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

#4. Preapare data

#Fetaures and target
x = df['X1', 'X2'] # select the 'X1' and 'X2' columns as features
y = df['Y'] # select the 'Y' column as the target variable

#Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Train the MLR model

#Initialize and train
model = LinearRegression() # create an instance of the LinearRegression class
model.fit(X_train, y_train)

#Cofficients
print("Intercept: ", model.intercept_) # This is the expected value of Y when both X1 and X2 are 0
print("Coefficients: ", model.coef_) #This shows how much Y is expected to change when X1 or X2 increases by 1 unit, holding the other variable constant

#6. Evaluate the model

#Predictions
y_pred = model.predict(X_test)

#Metrics
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R-squared: ", r2_score(y_test, y_pred))

#7. Visualize the results

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

#8. Make predictions on new data

new_data = pd.DataFrame({'X1': [5], 'X2': [2]})
prediction = model.predict(new_data)
print("Prediction for new data: ", prediction)