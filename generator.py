import random

n = random.randint(1000, 10000)
random_string = ''.join([str(random.randint(0, 1)) for _ in range(n)])


with open('results.txt', 'w') as f:
    f.write(random_string)
