import random
import os


def random_string():
    """_summary_
    """    
    n = random.randint(1000, 10000)
    random_string = "".join([str(random.randint(0, 1)) for _ in range(n)])
    return random_string


def __test__():
    rnd_string = random_string()

    # Get the path of the root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the subfolder containing the datasets
    subfolder_name = "dataset"

    # Construct the path to the file
    file_path = os.path.join(script_dir, "..", subfolder_name, "results.txt")

    with open(file_path, "w") as f:
        f.write(rnd_string)


if __name__ == "__main__":
    __test__()
