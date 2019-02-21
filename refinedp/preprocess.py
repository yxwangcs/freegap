import numpy as np


def process_frequent_itemsets(in_file, out_file):
    itemsets = []
    with open(in_file, 'r') as in_f:
        for line in in_f.readlines():
            for ch in line.split(','):
                if ch == '\n':
                    continue
                else:
                    itemsets.append(ch)
    items = np.asarray(itemsets, dtype=np.int)
    itemsets = np.unique(np.asarray(items), return_counts=True)[1]
    return items, itemsets
