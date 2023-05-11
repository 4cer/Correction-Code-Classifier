from reedsolo import RSCodec, ReedSolomonError
from bittobytes import bitstring_to_bytes
import itertools
import more_itertools


def reed_coder(data):
    rsc = RSCodec(10)
    final_string = ""
    for batch in more_itertools.chunked(data, 128):
       # print(batch)
        batch = ''.join(map(str, batch))
        #print(len(batch))
        data_byte = bitstring_to_bytes(batch)
        #print(data_byte)
        encoded_data = rsc.encode(data_byte)
        part_data = ''.join(format(byte, '08b') for byte in encoded_data)
        #print(len(part_data))
        final_string += part_data
    return final_string


def reed_decoder(data):
    rsc = RSCodec(10)
    for batch in more_itertools.chunked(data, 208):
        batch = ''.join(map(str, batch))
        data_byte = bitstring_to_bytes(batch)
        #print(rsc.decode(data_byte))
