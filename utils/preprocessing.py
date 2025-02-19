# preprocessing.py
import pandas as pd

def load_and_scale_data(filepath):
    """
    Returns a scaled version of the 
    input and target data points
    input: filepath of the data
    return: scaled input feature (X) and target (y)
    """
    df = pd.read_csv(filepath)
    
    # Calculate mean and std for scaling
    MEAN_X1 = df["shower_depth"].mean()
    SIGMA_X1 = df["shower_depth"].std()
    MEAN_X2 = df["shower_width"].mean()
    SIGMA_X2 = df["shower_width"].std()

    # Scale the input features
    df["shower_depth_scaled"] = (df["shower_depth"] - MEAN_X1) / SIGMA_X1
    df["shower_width_scaled"] = (df["shower_width"] - MEAN_X2) / SIGMA_X2

    # Add bias term (x0 = 1)
    df["x0"] = 1

    # Define input features
    X = df[["x0", "shower_depth_scaled", "shower_width_scaled"]]

    # Create target variable
    df["y"] = (df["type"] == "electron").astype(int)
    y = df["y"]

    return  X.to_numpy(), y.to_numpy().reshape(-1, 1)
