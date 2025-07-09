import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegressionCustom:
    def __init__(self, alpha=0.1, epochs=10000, tol=1e-6, patience=50, plot_cost=False):
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        self.patience = patience
        self.cost_history = []
        self.intercept_ = None
        self.coef_ = None
        self.mean_ = None
        self.std_ = None
        self.plot_cost = plot_cost

    def _standardize(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return (X - self.mean_) / self.std_

    def _standardize_infer(self, X):
        return (X - self.mean_) / self.std_

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, Y, w):
        m = len(Y)
        Y_pred = self.sigmoid(X @ w)
        Y_pred = np.clip(Y_pred, 1e-10, 1 - 1e-10)
        cost = -(1 / m) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
        return cost

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        X = self._standardize(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.zeros(X.shape[1])

        best_cost = float('inf')
        patience_counter = 0

        for i in range(self.epochs):
            Z = X @ w
            A = self.sigmoid(Z)
            gradient = (X.T @ (A - Y)) / len(Y)
            w -= self.alpha * gradient

            cost = self.compute_cost(X, Y, w)
            self.cost_history.append(cost)

            if best_cost - cost < self.tol:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {i}")
                    break
            else:
                best_cost = cost
                patience_counter = 0

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        if self.plot_cost:
            self._plot_cost()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        X = self._standardize_infer(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X @ np.r_[self.intercept_, self.coef_])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _plot_cost(self):
        plt.plot(self.cost_history)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
