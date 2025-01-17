# Ask ChatGPT to generate the python code for plotting the logisic regression function


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Features
y = (X > 5).astype(int).ravel()   # Labels (1 if X > 5, else 0)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Create a range of values for plotting the logistic function
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_plot)[:, 1]  # Get probabilities for the positive class

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X_plot, y_prob, color='blue', linewidth=2, label='Logistic Regression Curve')
plt.title('Logistic Regression Function')
plt.xlabel('Feature (X)')
plt.ylabel('Probability of Class 1')
plt.axhline(0.5, color='gray', linestyle='--')  # Threshold line
plt.axvline(5, color='green', linestyle='--', label='Decision Boundary (X=5)')
plt.legend()
plt.grid()
plt.show()
