import numpy as np


def byte_to_bits(byte: str):
    """ Converts a byte into a BigEndian-ordered array of ints with {0, 1} values

    Args:
        byte (str): A string of length 1, containing any ASCII character

    Raises:
        ValueError: If provided string has a value not of length 1

    Returns:
        _type_: List of integer values {0, 1} representing BigEndian-ordered bits of input
    """    
    if byte.__len__() != 1:
        raise ValueError(f'A string of length 1 expected')
    bits = []
    byte_int = ord(byte[0])
    for i in range(8):
        bit = 1 if byte_int & 128 else 0
        bits.append(bit)
        byte_int = byte_int << 1
    return bits


def string_to_bits(bytes: str):
    """ Converts a string of characters into a concatenated array of their BigEndian-ordered bits encoded as ints

    Args:
        bytes (str): A string of bytes to decode

    Returns:
        _type_: List of integer values {0, 1} represented concatenated bits of all input bytes in given order
    """    
    bits = []
    for i in range(bytes.__len__()):
        bits = bits + byte_to_bits(bytes[i])
    return bits


def test_noise(vector_length = 64):
    noise_array = np.random.rand(1, vector_length)
    with np.nditer(noise_array, op_flags=['readwrite']) as na:
        for s in na:
            bit_value = 1 if s > 0.5 else 0
            s[...] = bit_value
    return noise_array


def main():
    print(test_noise())


if __name__ == "__main__":
    main()