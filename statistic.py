cnt = {str(n): 0 for n in range(23)}

with open('test.txt') as f:
    lines = f.readlines()

for line in lines:
    anns = line.strip().split()[1:]
    for ann in anns:
        class_no = ann.split(',')[-1]
        cnt[class_no] += 1

for cls in cnt:
    print(cls, cnt[cls])
