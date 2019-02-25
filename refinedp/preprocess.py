import numpy as np


def _process(in_file, delimiter=' '):
    itemsets = []
    with open(in_file, 'r') as in_f:
        for line in in_f.readlines():
            for ch in line.split(delimiter):
                if ch == '\n':
                    continue
                else:
                    itemsets.append(ch)
    itemsets = np.unique(np.asarray(np.asarray(itemsets, dtype=np.int)), return_counts=True)
    return itemsets[1]


# helper functions
def process_bms_pos(in_file):
    return _process(in_file, ',')


def process_kosarak(in_file):
    return _process(in_file)
