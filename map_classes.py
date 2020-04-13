trans = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
         8: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 5,
         15: 6, 16: 7, 17: 6, 18: 6, 19: 8, 20: 9, 21: 10}

with open('annotations.txt') as f:
    lines = f.readlines()

with open('temp.txt', 'w') as f:
    for line in lines:
        anns = line.strip().split()
        path = anns[0]
        anns = anns[1:]
        boxes = []
        for ann in anns:
            box = [int(n) for n in ann.split(',')]
            if box[-1] in trans:
                box[-1] = trans[box[-1]]
                boxes.append(box)
        if boxes:
            f.write(path)
            for box in boxes:
                f.write(' %d,%d,%d,%d,%d' % tuple(box))
            f.write('\n')
