import bchlib
import hashlib
import os
import random
import itertools
import more_itertools
from bittobytes import bitstring_to_bytes

def bch_coder(data):
    # create a bch object
    BCH_POLYNOMIAL = random.choice([8219, 16427, 32771])
    BCH_BITS = random.randint(8,25)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    final_string = ""
    # random data
    for batch in more_itertools.chunked(data, 128):
        batch = ''.join(map(str, batch))
        data_byte = bitstring_to_bytes(batch)
        # encode and make a "packet"
        ecc = bch.encode(data_byte)
        bytes_as_bits = ''.join(format(byte, '08b') for byte in ecc)
        packet = data + bytes_as_bits
        a = ''.join(map(str, packet))
        final_string += a
    return final_string
