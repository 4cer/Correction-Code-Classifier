import bchlib
import hashlib
import os
import random
import itertools
import more_itertools
from bittobytes import bitstring_to_bytes

def bch_coder(data):
    # create a bch object
    BCH_POLYNOMIAL = 8219
    BCH_BITS = 10
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
        final_string += a;
    return final_string

def bch_decoder(data):

    # de-packetize
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

    # correct
    bitflips = bch.decode_inplace(data, ecc)
    print('bitflips: %d' % (bitflips))

    # packetize
    packet = data + ecc

    # print hash of packet
    sha1_corrected = hashlib.sha1(packet)
    print('sha1: %s' % (sha1_corrected.hexdigest(),))

    if sha1_initial.digest() == sha1_corrected.digest():
        print('Corrected!')
    else:
        print('Failed')