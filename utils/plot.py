#plot.py
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(costs_train, costs_test, N):
    """Plot training and testing loss over iterations."""
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(N), costs_train, color="orangered", label="Train set", lw=1)
    plt.plot(np.arange(N), costs_test, color="blue", label="Test set", lw=1)
    plt.xlabel("Iterations")
    plt.ylabel(r"Cost $J(\theta)$")
    plt.title("Cross Entropy Loss over Iterations", fontsize=18, pad=20)
    plt.legend()
    plt.show()

def rev_sigmoid(y):
    return np.log(y / (1 - y))

def scale_inputs(input_feature, mu, sigma):
    return (input_feature - mu) / sigma

def unscale_inputs(scaled_feature, mu, sigma):
    return scaled_feature * sigma + mu

def plot_decision_boundary(sigs, bkgs, thresholds, thetas, x1_min=0, x1_max=200, x2_min=0, x2_max=60, 
                           mu_d=0, sigma_d=1, mu_w=0, sigma_w=1):
    """Plot decision boundary for different thresholds."""
    if thresholds is None:  # Default threshold if not provided
        thresholds = [0.5]

    theta0, theta1, theta2 = thetas

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot for signals and backgrounds
    ax.scatter(sigs[:, 0], sigs[:, 1], alpha=0.2, s=1, color="dodgerblue", label="electron")
    ax.scatter(bkgs[:, 0], bkgs[:, 1], alpha=0.2, s=1, color="darkorange", label="hadron")

    ax.set_xlim([x1_min, x1_max])
    ax.set_ylim([x2_min, x2_max])
    ax.set_xlabel("Shower Depth [mm]")
    ax.set_ylabel("Shower Width [mm]")
    ax.set_title("Calorimeter Showers")

    # Plot boundary lines for each threshold
    for t in thresholds:
        boundary_val = rev_sigmoid(t)

        # Scale x1 (shower depth) points
        x1_left_scaled = scale_inputs(x1_min, mu_d, sigma_d)
        x1_right_scaled = scale_inputs(x1_max, mu_d, sigma_d)

        # Compute corresponding x2 (shower width) values based on the threshold
        x2_left_scaled = (boundary_val - theta0 - theta1 * x1_left_scaled) / theta2
        x2_right_scaled = (boundary_val - theta0 - theta1 * x1_right_scaled) / theta2

        # Unscale for plotting
        x2_left_unscaled = unscale_inputs(x2_left_scaled, mu_w, sigma_w)
        x2_right_unscaled = unscale_inputs(x2_right_scaled, mu_w, sigma_w)

        
        line_style = "-" if t == 0.5 else "--"
        ax.plot([x1_min, x1_max], [x2_left_unscaled, x2_right_unscaled], label=f"$\\hat{{y}}={t}$", ls=line_style)

    ax.legend()
    plt.show()
