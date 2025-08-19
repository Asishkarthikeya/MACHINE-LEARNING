import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression (simplified)
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        
        for _ in range(self.epochs):
            # Linear model
            z = np.dot(X, self.w) + self.b
            h = sigmoid(z)
            
            # Gradients
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return (sigmoid(z) >= 0.5).astype(int)

# ---------------- DRIVER CODE ---------------- #

# Generate simple dataset
np.random.seed(0)
X = np.random.rand(200, 2) * 10   # 200 samples, 2 features
y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Label = 1 if x1+x2>10

# Train/Test split (manual, no sklearn)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Accuracy: {acc:.2f}")
