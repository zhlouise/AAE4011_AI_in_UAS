import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP

import time

# Generate synthetic dataset with two moons (non - linearly separable )
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    # Create a mesh grid
    h = 0.02 # step size in the mesh
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions on the mesh grid
    if isinstance(model, MLP):
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points).reshape(xx.shape)
    elif isinstance(model, LinearRegression):
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.round(model.predict(mesh_points)).clip(0, 1).reshape(xx.shape)
    else: # LogisticRegression
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points).reshape(xx.shape)

    # Plot decision boundary and training points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors ='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 1. Linear Regression ( treating classification as regression )
print("Training Linear Regression model ...")
lr_model = LinearRegression()
start_time = time.time()
lr_history = lr_model.fit(X_train_scaled, y_train)
lr_train_time = time.time()-start_time
lr_preds = np.round(lr_model.predict(X_test_scaled)).clip(0, 1)
lr_accuracy = np.mean(lr_preds == y_test)
print(f"Linear Regression - Accuracy: {lr_accuracy:.4f}, Training Time: {lr_train_time:.4f}s")

# 2. Logistic Regression
print("\nTraining Logistic Regression model ...")
log_model = LogisticRegression()
start_time = time.time()
log_history = log_model.fit(X_train_scaled, y_train)
log_train_time = time.time() - start_time
log_preds = log_model.predict(X_test_scaled)
log_accuracy = np.mean(log_preds == y_test)
print(f"Logistic Regression - Accuracy: {log_accuracy:.4f}, Training Time: {log_train_time:.4f}s")

# 3. Multi - Layer Perceptron
print("\nTraining MLP model ...")
# One hidden layer with 10 neurons , ReLU activation , sigmoid output
mlp_model = MLP(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
start_time = time.time()
X_train_mlp = X_train_scaled.T
y_train_mlp = y_train.reshape(1, -1)
X_test_mlp = X_test_scaled.T
mlp_history = mlp_model.fit(X_train_scaled, y_train)
mlp_train_time = time.time () - start_time

# Get MLP predictions
mlp_preds = mlp_model.predict(X_test_scaled)
mlp_accuracy = np.mean(mlp_preds == y_test)
print(f"MLP - Accuracy: {mlp_accuracy:.4f}, Training Time: { mlp_train_time:.4f}s")

# Plot decision boundaries
plot_decision_boundary(lr_model, X_test_scaled, y_test, 'Linear Regression Decision Boundary')
plot_decision_boundary(log_model, X_test_scaled, y_test, 'Logistic Regression Decision Boundary')
plot_decision_boundary(mlp_model, X_test_scaled, y_test, 'MLP Decision Boundary')