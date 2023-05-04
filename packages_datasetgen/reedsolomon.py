from reedsolo import RSCodec, ReedSolomonError
import itertools
import more_itertools

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

def reed_codec(data):
    rsc = RSCodec(10)
    final_string = ""
    for batch in more_itertools.chunked(data, 128):
        print(batch)
        batch = ''.join(map(str, batch))
        print(len(batch))
        data_byte = bitstring_to_bytes(batch)
        print(data_byte)
        encoded_data = rsc.encode(data_byte)
        part_data = ''.join(format(byte, '08b') for byte in encoded_data)
        print(len(part_data))
        final_string += part_data
    return final_string
def reed_decodec(data):
    rsc = RSCodec(10)
    for batch in more_itertools.chunked(data, 208):
        batch = ''.join(map(str, batch))
        data_byte = bitstring_to_bytes(batch)
        print(rsc.decode(data_byte))
