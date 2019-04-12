import logging
import csv
import numpy as np


logger = logging.getLogger(__name__)


def _process(in_file, delimiter=' '):
    itemsets = []
    records = 0
    with open(in_file, 'r') as in_f:
        for line in in_f.readlines():
            for ch in line.split(delimiter):
                if ch == '\n':
                    records += 1
                    continue
                else:
                    itemsets.append(ch)
    itemsets = np.unique(np.asarray(np.asarray(itemsets, dtype=np.int)), return_counts=True)
    logger.info('Statistics for {}: # of records: {} and # of Items: {}'.format(in_file, records + 1, len(itemsets[0])))
    res = itemsets[1]
    np.random.RandomState(0).shuffle(res)
    return res


# helper functions
def process_bms_pos(in_file):
    return _process(in_file, ',')


def process_kosarak(in_file):
    return _process(in_file)


def process_t40100k(in_file):
    return _process(in_file)


def process_sf1(in_file):
    with open(in_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = None
        for row_index, row in enumerate(reader):
            if row_index == 1:
                data = row
        data = np.asarray(tuple(map(lambda cell: int(cell), data[3:])))
        logger.info('Statistics for {}:  # of Items: {}'.format(in_file, len(data)))
        np.random.RandomState(0).shuffle(data)
        return data

