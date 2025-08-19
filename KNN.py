import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# KNN from scratch
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        preds = []
        for x in X_test:
            preds.append(self._predict_one(x))
        return np.array(preds)
    
    def _predict_one(self, x):
        # Euclidean distance to all training points
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Indices of k nearest
        k_idx = np.argsort(distances)[:self.k]
        # Labels of neighbors
        k_labels = self.y_train[k_idx]
        # Majority vote
        return Counter(k_labels).most_common(1)[0][0]

# ---------------- DRIVER CODE ---------------- #

def main():
    # Load dataset
    df = pd.read_csv("/Users/asishkarthikeyagogineni/Desktop/ML/diabetes.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    # Our KNN
    my_knn = KNN(k=3)
    my_knn.fit(X_train, y_train)
    preds = my_knn.predict(X_test)

    # Sklearn KNN
    skl_knn = KNeighborsClassifier(n_neighbors=3)
    skl_knn.fit(X_train, y_train)
    preds_skl = skl_knn.predict(X_test)

    # Accuracy
    acc_my = np.mean(preds == y_test) * 100
    acc_skl = np.mean(preds_skl == y_test) * 100

    print(f"Our KNN Accuracy     : {acc_my:.2f}%")
    print(f"Sklearn KNN Accuracy : {acc_skl:.2f}%")

if __name__ == "__main__":
    main()
