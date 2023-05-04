from hamming import hamming_codec, hamming_decodec
import os
import random
data = bytearray(os.urandom(1024))
d = ''.join(format(byte, '08b') for byte in data)
coded = hamming_codec(d)
hamming_decodec(coded)
