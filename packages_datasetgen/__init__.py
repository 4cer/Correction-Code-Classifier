from hamming import hamming_coder, hamming_decoder
from reedsolomon import reed_coder, reed_decoder
from bch import bch_coder
import os
import random

i = 0
f = open("demofile.txt", "w")
while i < 10000:
    data_byte = bytearray(os.urandom(256))
    data_bit = ''.join(format(byte, '08b') for byte in data_byte)
    rnd = random.randint(0, 3)
    coded = ""
    if rnd == 0:
        data_bit = data_bit[:382]
        f.write(data_bit + ";0\n")
    if rnd == 1:
        coded = hamming_coder(data_bit)
        hamming_decoder(coded)
        coded = coded[:382]
        f.write(coded + ";1\n")
    if rnd == 2:
        coded = bch_coder(data_bit)
        # bch_decoder(coded)
        coded = coded[:382]
        f.write(coded + ";2\n")
    if rnd == 3:
        coded = reed_coder(data_bit)
        reed_decoder(coded)
        coded = coded[:382]
        f.write(coded + ";3\n")

    i += 1
f.close()
