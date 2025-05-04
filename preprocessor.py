import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
load_and_process
Load the Iris dataset from a text file, normalize features, one-hot encode labels,
and split into training, validation, and test sets using scikit.
Takes in the following: 
    data_path (str): path to the text file containing the Iris data.
    test_size (float): proportion of data reserved for testing + validation.
    val_size (float): proportion of data reserved for validation (relative to test+val).
    random_state (int): seed for reproducibility.
** NOTE: By default, the preprocessor does a 60/20/20 split for training/validation/testing ** 
Returns a tuple containing two arrays for each subset of the data (one for the features,
and another for the labels corresponding to those entries)-- so six arrays in total.
"""
def load_and_preprocess(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
]:
    # Read in the dataset as a csv assuming last col is the label
    features: list[float] = []  
    labels: list[str] = []
    with open(data_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # First four cols are features, last is label as string
            features.append([float(x) for x in row[:4]])
            labels.append(row[4])
    
    X = np.array(features)

    # Z-score normalization (0 mean and 1 stdev) w/ scikit
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Map text labels to integer indices
    unique_labels = sorted(set(labels))
    
    # Creates a dict where each string label is associated with an index
    # (e.g {Iris-setosa: 0, Iris-versicolor: 1, Iris-virginica: 2})
    label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y = np.array([label_to_int[lbl] for lbl in labels], dtype=int)

    # One-hot encode labels
    num_classes = len(np.unique(y))
    # Create an array of [0,0,0] arrays
    Y = [[0] * num_classes for _ in range(len(y))]
    # Set the index of the label "on" to represent each label as 1-hot
    for i, label in enumerate(y):
        Y[i][label] = 1
    # Convert to numpy
    Y = np.array(Y)

    # NOTE: Partitioning proportions are submitted
    # Stratified train/val/test split using sklearn's train_test_split which
    # takes a proporiton of the dataset and saves it into X/Y_train
    # Leaving the remaining data entries in the temp vars.
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_scaled, Y,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=random_state
    )
    # Calculate the remaining proportion to take for the validation set
    val_relative = val_size / (test_size + val_size)
    # Save the remainder from the partitioning as the test set.
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=val_relative,
        stratify=np.argmax(Y_temp, axis=1),
        random_state=random_state
    )
    # Return the 3 partitions with Y being the labels and X as features.
    return X_train, Y_train, X_val, Y_val, X_test, Y_test