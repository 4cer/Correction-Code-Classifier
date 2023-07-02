import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(filename, skiprows=None, nrows=None):
    # Read the CSV file
    data = pd.read_csv(filename, sep=';', header=None, skiprows=skiprows, nrows=nrows)
    
    # Convert binary strings to integer arrays
    data[0] = data[0].apply(lambda x: [int(bit) for bit in x])

    # Convert label to 4 binary columns
    # data[1] = [binarize(row) for row in data[1]]
    
    # Split the data into features (X) and labels (y)
    X = data[0].tolist()
    y = data[1].tolist()
    
    # Split the data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.263, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test