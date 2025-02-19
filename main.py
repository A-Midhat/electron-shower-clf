# main.py
from utils.preprocessing import load_and_scale_data
from model.log_reg import LogisticRegression, Evaluate
from utils.plot import plot_loss, plot_decision_boundary
from sklearn.model_selection import train_test_split

# Preprocess data
X, y = load_and_scale_data("./data/dataset.csv") 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(lr=0.05, iterations=1000)
best_thetas, costs_train, costs_test = model.fit(X_train, y_train, X_test, y_test)

# Evaluate
eval = Evaluate(model)
y_pred = eval.make_predictions(X_test)  # No need for `best_thetas`
accuracy = eval.get_accuracy(y_test, y_pred)
recall = eval.get_recall(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}%")
print(f"Model Recall: {recall:.2f}")

# Plot losses
plot_loss(costs_train, costs_test, N=len(costs_train))

# Decision boundary plot
sigs = X[y[:, 0] == 1]  
bkgs = X[y[:, 0] == 0]  
mu_d, sigma_d = X[:, 1].mean(), X[:, 1].std()  
mu_w, sigma_w = X[:, 2].mean(), X[:, 2].std()

thresholds = [0.10, 0.25, 0.5, 0.75, 0.90]
plot_decision_boundary(sigs, bkgs, thresholds, best_thetas.flatten(), mu_d, sigma_d, mu_w, sigma_w)
