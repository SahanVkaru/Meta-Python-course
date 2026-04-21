import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# Exercise 1: Build an SLR Model on Synthetic Data
# ==========================================
print("\n--- Exercise 1: SLR on Synthetic Data ---")
# 1. Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Visualize the data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Points')
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Target Variable)")
plt.title("Exercise 1: Synthetic Data Scatter Plot")
plt.legend()
plt.show()

# 3. Fit the model
slr_model = LinearRegression()
slr_model.fit(X, y)

# 4. Print intercept and coefficient
print(f"Intercept: {slr_model.intercept_[0]:.4f}")
print(f"Coefficient: {slr_model.coef_[0][0]:.4f}")

# 5. Predict on new x values
X_new = np.array([[0], [2]])
y_predict = slr_model.predict(X_new)
print(f"Predictions for X=[0, 2]:\n{y_predict}")

# 6. Plot regression line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue')
plt.plot(X_new, y_predict, color='red', linewidth=2, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Exercise 1: SLR Regression Line")
plt.legend()
plt.show()

# 7. Calculate MSE and R2
y_pred_all = slr_model.predict(X)
print(f"Mean Squared Error (MSE): {mean_squared_error(y, y_pred_all):.4f}")
print(f"R-squared (R2): {r2_score(y, y_pred_all):.4f}")


# ==========================================
# Exercise 2: SLR on Real Dataset (California Housing)
# ==========================================
# Activity 2: Using a real dataset (California Housing is used as Boston is deprecated)
print("\n--- Exercise 2: SLR on California Housing ---")
# 1. Load dataset
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['MedHouseVal'] = housing.target

# 2. Select 'MedInc' as independent variable
X_medinc = df_housing[['MedInc']]
y_house = df_housing['MedHouseVal']

# 3. Fit SLR model
slr_housing = LinearRegression()
slr_housing.fit(X_medinc, y_house)

# 4. Visualize using regression plot
plt.figure(figsize=(8, 5))
sns.regplot(x='MedInc', y='MedHouseVal', data=df_housing.sample(500), scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Exercise 2: Median Income vs House Value (Sampled)")
plt.show()

# 5. Evaluate performance
y_pred_housing = slr_housing.predict(X_medinc)
print(f"R-squared for MedInc: {r2_score(y_house, y_pred_housing):.4f}")


# ==========================================
# Exercise 3: MLR on Synthetic Dataset
# ==========================================
print("\n--- Exercise 3: MLR on Synthetic Dataset ---")
# 1. Generate synthetic data
X1 = np.random.rand(100) * 10
X2 = np.random.rand(100) * 5
# Activity: Add X3 (random noise)
X3 = np.random.randn(100) 
y_mlr = 3 + 2*X1 + 4*X2 + 0.5*X3 + np.random.randn(100)

# 2. Create DataFrame
df_mlr = pd.DataFrame({'x1': X1, 'x2': X2, 'x3': X3, 'y': y_mlr})

# 3. Train model (using x1 and x2 first as per Exercise 3 steps)
X_train_mlr = df_mlr[['x1', 'x2']]
y_train_mlr = df_mlr['y']
mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)

# 4. Display coefficients and intercept
print(f"Intercept: {mlr_model.intercept_:.4f}")
print(f"Coefficients (x1, x2): {mlr_model.coef_}")

# 5. Plot 3D surface
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_mlr['x1'], df_mlr['x2'], df_mlr['y'], color='blue', alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.title("Exercise 3: 3D Visualization of MLR Data")
plt.show()

# 6. Evaluate
y_pred_mlr = mlr_model.predict(X_train_mlr)
print(f"MSE: {mean_squared_error(y_train_mlr, y_pred_mlr):.4f}")
print(f"R2: {r2_score(y_train_mlr, y_pred_mlr):.4f}")


# ==========================================
# Exercise 4: MLR with Feature Selection
# ==========================================
print("\n--- Exercise 4: MLR with Feature Selection ---")
X_cal = df_housing.drop('MedHouseVal', axis=1)
y_cal = df_housing['MedHouseVal']

X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

model_cal = LinearRegression()
model_cal.fit(X_train_cal, y_train_cal)

# Display coefficients
coef_df = pd.DataFrame({'Feature': X_cal.columns, 'Coefficient': model_cal.coef_})
print("Coefficients:\n", coef_df)

# Rank features by absolute coefficient values
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
print("\nRanked Features:\n", coef_df.sort_values(by='Abs_Coef', ascending=False))

# Use statsmodels to get p-values
X_train_sm = sm.add_constant(X_train_cal)
ols_model = sm.OLS(y_train_cal, X_train_sm).fit()
print("\nStatsmodels OLS Summary Highlights (p-values):")
print(ols_model.pvalues)


# ==========================================
# Exercise 5: Compare SLR vs MLR
# ==========================================
print("\n--- Exercise 5: Compare SLR vs MLR ---")
# SLR (MedInc)
slr_comp = LinearRegression()
slr_comp.fit(X_train_cal[['MedInc']], y_train_cal)
y_pred_slr = slr_comp.predict(X_test_cal[['MedInc']])

# MLR (All features)
mlr_comp = LinearRegression()
mlr_comp.fit(X_train_cal, y_train_cal)
y_pred_mlr_cal = mlr_comp.predict(X_test_cal)

print(f"SLR R2: {r2_score(y_test_cal, y_pred_slr):.4f} | MSE: {mean_squared_error(y_test_cal, y_pred_slr):.4f}")
print(f"MLR R2: {r2_score(y_test_cal, y_pred_mlr_cal):.4f} | MSE: {mean_squared_error(y_test_cal, y_pred_mlr_cal):.4f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test_cal, y_pred_mlr_cal, alpha=0.3, label='MLR Predictions')
plt.plot([min(y_test_cal), max(y_test_cal)], [min(y_test_cal), max(y_test_cal)], '--r', label='Ideal')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Exercise 5: Actual vs Predicted (MLR)")
plt.legend()
plt.show()


# ==========================================
# Exercise 6: Visualizing MLR Residuals
# ==========================================
print("\n--- Exercise 6: Visualizing MLR Residuals ---")
residuals = y_test_cal - y_pred_mlr_cal

plt.figure(figsize=(12, 5))

# Histogram/KDE
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")

# Residuals vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_pred_mlr_cal, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Heteroscedasticity check)")

plt.tight_layout()
plt.show()


# ==========================================
# Exercise 7: Use Pipeline and Scaling
# ==========================================
print("\n--- Exercise 7: Pipeline and Scaling ---")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train_cal, y_train_cal)
y_pred_scaled = pipeline.predict(X_test_cal)
print(f"Scaled Pipeline R2: {r2_score(y_test_cal, y_pred_scaled):.4f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_cal, y_cal, cv=5)
print(f"Cross-val Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")


# ==========================================
# Activities: Normal Equation & Polynomial Regression
# ==========================================
print("\n--- Additional Activities ---")

# 4. Manually calculate coefficients using the Normal Equation: theta = (X.T * X)^-1 * X.T * y
print("\nActivity 4: Normal Equation")
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"Normal Equation Intercept: {theta_best[0][0]:.4f}")
print(f"Normal Equation Coefficient: {theta_best[1][0]:.4f}")

# 5. Compare with Polynomial Regression
print("\nActivity 5: Polynomial Regression (Degree 2)")
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)
print(f"Polynomial Regression R2: {r2_score(y, y_poly_pred):.4f}")
