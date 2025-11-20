"""
Hey feel free to learn with me...
Machine Learning Project: k-Nearest Neighbors (kNN) Classification
This script implements a kNN classifier on the MAGIC gamma telescope dataset
to classify gamma rays vs hadrons.
"""

# Data manipulation and analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Define column names for the MAGIC dataset
# Features: fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist
# Target: class (gamma 'g' or hadron 'h')
cols = ["fLenght", "fWidth", "fSize", "fConc", "fConc1", "fAsym", 
        "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

# Load the dataset
# Note: Make sure "magic04.data" is in the same directory as this script
df = pd.read_csv("magic04.data", names=cols)

# Display first few rows to understand the data structure
print("First few rows of the dataset:")
print(df.head())

# Convert class labels from 'g' (gamma) and 'h' (hadron) to binary: 1 for gamma, 0 for hadron
df["class"] = (df["class"] == "g").astype(int)

print("\nDataset after converting class labels:")
print(df.head())

# Visualize the distribution of each feature for both classes
# This helps understand how features differ between gamma and hadron events
print("\nGenerating histograms for feature analysis...")
for label in cols[:-1]:  # Exclude the 'class' column
    plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", 
             alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color="red", label="hadron", 
             alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Split the dataset into training (60%), validation (20%), and test (20%) sets
# Shuffle the data first to ensure random distribution
print("\nSplitting dataset into train/validation/test sets...")
train, valid, test = np.split(df.sample(frac=1), 
                               [int(.6*len(df)), int(.8*len(df))])

print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(valid)}")
print(f"Test set size: {len(test)}")


def scale_dataset(dataframe, oversample=False):
    """
    Scale the features and optionally apply oversampling to handle class imbalance.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The dataset to scale
    oversample : bool, default=False
        Whether to apply random oversampling to balance classes
    
    Returns:
    --------
    data : numpy.ndarray
        Combined scaled features and labels
    X : numpy.ndarray
        Scaled feature matrix
    y : numpy.ndarray
        Target labels
    """
    # Extract features (all columns except the last one)
    X = dataframe[dataframe.columns[:-1]].values
    
    # Extract target labels (last column)
    y = dataframe[dataframe.columns[-1]].values
    
    # Standardize features: mean=0, std=1
    # This is important for distance-based algorithms like kNN
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply random oversampling if requested (useful for imbalanced datasets)
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    # Combine scaled features and labels back into a single array
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y


# Scale and prepare the datasets
# Apply oversampling to training set to handle class imbalance
print("\nScaling datasets...")
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# Initialize and train the k-Nearest Neighbors classifier
# k=5 means we look at the 5 nearest neighbors to make predictions
print("\nTraining kNN model (k=5)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
print("\nMaking predictions on test set...")
y_pred = knn_model.predict(X_test)

# Evaluate the model performance
print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(y_test, y_pred))
print("="*50)
