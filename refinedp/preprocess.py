import numpy as np


def process_frequent_itemsets(in_file, out_file):
    itemsets = []
    with open(in_file, 'r') as in_f:
        for line in in_f.readlines():
            itemsets.extend(line.split())

    itemsets = np.unique(np.asarray(itemsets), return_counts=True)[1]
    np.save(out_file, itemsets)
