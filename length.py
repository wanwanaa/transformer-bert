import numpy as np


if __name__ == '__main__':
    filename_text = 'DATA/raw_data/train.source'
    l = []
    with open(filename_text, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            l.append(len(line))
    mean = sum(l) / len(l)
    # print(mean)
    var = np.std(l)
    result = mean + var
    result = int(round(result))
    print('max:', max(l))      # test:140  # summary:30
    print('min', min(l))       # test:80   # summary:8
    print('mean:', mean)       # test:103  # summary:17
    print('mean+var:', result) # test:114  # summary:22