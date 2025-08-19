import numpy as np
import matplotlib.pyplot as plt

# Sample data (YearsExperience vs Salary-like data)
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
m = 0   # slope
b = 0   # intercept
lr = 0.01   # learning rate
epochs = 1000   # iterations

n = len(X)

# Gradient Descent
for _ in range(epochs):
    Y_pred = m*X + b
    # Gradients
    dm = (-2/n) * sum(X * (Y - Y_pred))
    db = (-2/n) * sum(Y - Y_pred)
    # Update
    m -= lr * dm
    b -= lr * db

print(f"Trained slope (m): {m:.2f}")
print(f"Trained intercept (b): {b:.2f}")

# Predictions
Y_pred = m*X + b

# Plot
plt.scatter(X, Y, color='blue')
plt.plot(X, Y_pred, color='orange')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.show()
