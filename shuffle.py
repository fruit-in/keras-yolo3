import random

with open('temp.txt') as f:
    lines = f.readlines()

random.shuffle(lines)

with open('temp.txt', 'w') as f:
    for line in lines:
        f.write(line)
