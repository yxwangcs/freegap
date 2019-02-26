import logging
import os
import numpy as np
from refinedp.algorithms import sparse_vector, laplace_mechanism
from refinedp.preprocess import process_bms_pos, process_kosarak


logger = logging.getLogger(__name__)


def gap_sparse_vector(q, threshold, c, epsilon):
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < c:
        eta_i = np.random.laplace(scale=4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(noisy_q_i - noisy_threshold)
            count += 1
        else:
            out.append(False)
        i += 1
    return out


def estimate_sparse_vector(q, threshold, c, epsilon):
    answers = sparse_vector(q, threshold, c, epsilon / 2.0)
    return np.nonzero(answers), laplace_mechanism(q, answers, epsilon / 2.0)


def evaluate_gap_sparse_vector(dataset_folder='datasets', output_folder='./figures/gap-sparse-vector'):
    logger.info('Evaluating Gap Sparse Vector')
    # create the output folder if not exists
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    path_prefix = os.path.abspath(output_folder)
    epsilon = 0.3

    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(dataset_folder)
    datasets = {
        'BMS-POS': process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder)),
        'kosarak': process_kosarak('{}/kosarak.dat'.format(dataset_folder))
    }

    c_array = list(range(25, 325, 25))

    for name, data in datasets.items():
        logger.info('Evaluating on {}'.format(name))
        sorted_data = np.sort(data)[::-1]
        for c in c_array:
            threshold = (sorted_data[c] + sorted_data[c + 1]) / 2.0
            r1 = np.asarray(gap_sparse_vector(data, threshold, c, epsilon))
            # filter out the gaps
            i1 = np.nonzero(r1)
            r1 = r1[i1] + threshold
            i2, r2 = estimate_sparse_vector(data, threshold, c, epsilon)
            r2 = np.asarray(r2)
            data = np.asarray(data)
            gap_err = np.sqrt(np.sum(np.square(data[i1] - r1)) / float(len(r1)))
            lap_err = np.sqrt(np.sum(np.square(data[i2] - r2)) / float(len(r2)))
            print('gap: {}, laplace: {}'.format(gap_err, lap_err))


