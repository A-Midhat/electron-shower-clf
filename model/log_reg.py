# log_reg.py
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.05, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.thetas = None

    # Linear summation (z)
    def lin_sum(self, X):
        return np.dot(X, self.thetas)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Hypothesis function h(z)
    def h_class(self, X):
        return self.sigmoid(self.lin_sum(X))

    def cross_entropy_cost(self, y_vals, y_preds):
        return -np.sum(y_vals * np.log(y_preds) + (1 - y_vals) * np.log(1 - y_preds)) / len(y_vals)

    def derivatives_cross_entropy(self, y_preds, y_obs, x_feature):
        return x_feature.T.dot(y_preds - y_obs) / len(y_obs)

    def fit(self, X_train, y_train, X_test, y_test):
        # Initialize thetas randomly
        self.thetas = np.random.randn(X_train.shape[1], 1)

        # Log for costs
        costs_train = []
        costs_test = []

        for i in range(1, self.iterations + 1):
            # Train
            y_preds_train = self.h_class(X_train)
            J_train = self.cross_entropy_cost(y_train, y_preds_train)
            costs_train.append(J_train)

            # Test
            y_preds_test = self.h_class(X_test)
            J_test = self.cross_entropy_cost(y_test, y_preds_test)
            costs_test.append(J_test)

            # Gradient and parameter update
            gradients = self.derivatives_cross_entropy(y_preds_train, y_train, X_train)
            self.thetas -= self.lr * gradients

            # Print updates (progress monitoring)
            if (i < 100 and i % 10 == 0) or (i > 100 and i % 100 == 0):
                print(f"[{i}] t0 = {self.thetas[0, 0]:.4f}, t1 = {self.thetas[1, 0]:.4f}, "
                      f"t2 = {self.thetas[2, 0]:.4f}, Cost = {J_train:.4f}, "
                      f"dJ0 = {gradients[0, 0]:.4f}, dJ1 = {gradients[1, 0]:.4f}, dJ2 = {gradients[2, 0]:.4f}")

        return self.thetas, costs_train, costs_test


class Evaluate:
    def __init__(self, model):
        self.model = model  # Store the model instance

    def make_predictions(self, X, boundary=0.5):
        # Use the learned thetas directly from the model
        y_preds_proba = self.model.h_class(X)
        bin_preds = (y_preds_proba >= boundary).astype(int)
        return bin_preds

    def get_accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        acc = correct_predictions / len(y_true) * 100  # Percentage accuracy
        return acc

    def get_recall(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        return recall
