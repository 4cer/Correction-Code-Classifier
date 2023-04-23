import numpy as np
import torch

import packages_classifier.example_module as em
import packages_gio.read_data as rd


def pytorch_test(nvidia = True):
    x = torch.rand(5, 3)
    print(x)
    if nvidia:
        print(f"CUDA available: {torch.cuda.is_available()}")


def module_test():
    input_string = chr(129) + chr(130) + chr(132)
    output_array = np.array([em.string_to_bits(input_string)])
    print(output_array)

    input_string = "abc"
    output_array2 = np.array([em.string_to_bits(input_string)])
    print(output_array2)
    
    print(np.zeros(dtype=np.uint, shape=(1, 24)))


def file_to_array_test():
    myArray = rd.read_file("results.txt")
    print(rd.read_range(myArray, 1, 300))


def main():
    pytorch_test()
    module_test()
    file_to_array_test()


if __name__ == "__main__":
    main()

