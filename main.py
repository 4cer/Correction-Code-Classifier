import numpy as np
import os

def readFile(file_name):

    # Get the path of the root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the subfolder containing the datasets
    subfolder_name = 'dataset'

    # Construct the path to the file
    file_path = os.path.join(script_dir, subfolder_name, file_name)

    # read the contents of the file
    with open(file_path, 'r') as f:
        content = f.read()

    # calculate the length of the string
    length = len(content)

    # convert the string to a NumPy array of floats
    data = np.array(list(content), dtype=np.float32)

    return data

def readRange(data, start_idx, n):
    # check if the range of indices is within the bounds of the array
    if start_idx < 0 or start_idx + n > data.shape[0]:
        raise IndexError('Index out of bounds')
    else:
        # create a slice of the array containing n consecutive elements starting from the given index
        slice = data[start_idx:start_idx + n]
        return slice


myArray = readFile("results.txt")
print(readRange(myArray, 1, 300))