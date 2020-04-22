import matplotlib.pyplot as plt

classes = range(11)
plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 900)

with open('../annotations.txt') as f:
    lines = f.readlines()
    w = []
    h = []
    for line in lines:
        line = line.strip().split()[1:]
        for x in line:
            x = [int(y) for y in x.split(',')]
            if x[4] in classes:
                w.append(x[2] - x[0])
                h.append(x[3] - x[1])
plt.scatter(w, h, alpha=.01, s=5)

plt.show()
