import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from refinedp.preprocess import process_bms_pos
from refinedp.algorithms import laplace_mechanism


logger = logging.getLogger(__name__)


def gap_noisy_max(q, epsilon):
    i, imax, max, gap = 0, 1, 0, 0
    while i < len(q):
        eta_i = np.random.laplace(scale=2.0 / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i > max or i == 1:
            imax = i
            gap = noisy_q_i - max
            max = noisy_q_i
        else:
            if noisy_q_i > max - gap:
                gap = max - noisy_q_i
    return imax, gap


def gap_k_noisy_max(q, k, epsilon):
    assert k <= len(q), 'k must be less or equal than the length of q'
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps


def naive_estimate(q, k, epsilon):
    # independently rerun naive approach
    indices, _ = gap_k_noisy_max(q, k, 0.5 * epsilon)
    estimates = laplace_mechanism(q, indices, 0.5 * epsilon)
    return indices, estimates


def refined_estimate(q, k, epsilon):
    indices, gaps = gap_k_noisy_max(q, k, 0.5 * epsilon)
    estimates = laplace_mechanism(q, indices, 0.5 * epsilon)
    coefficient = np.eye(k, k) * 3 + np.eye(k, k, 1) * -1 + np.eye(k, k, -1) * -1
    coefficient[0][0] = 2
    coefficient[k - 1][k - 1] = 2
    b = np.append(gaps, 0)
    b = b - np.roll(b, 1) + estimates
    final_estimates = np.linalg.solve(coefficient, b)
    return indices, final_estimates


def evaluate_gap_k_noisy_max(dataset_folder='datasets', output_folder='./figures/gap-noisymax'):
    logger.info('Evaluating Gap K Noisy Max')
    # create the output folder if not exists
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    path_prefix = os.path.abspath(output_folder)

    epsilon = 0.3
    datasets = {
        'BMS-POS': process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder))
    }

    k_array = range(25, 325, 25)

    for name, data in datasets.items():
        logger.info('Evaluating on {}'.format(name))
        metric_data, err_data = [[], []], [[], []]
        for k in k_array:
            results_1, results_2, results_3 = [], [], []
            for _ in range(10):
                i1, r1 = refined_estimate(data, k, epsilon)
                i2, r2 = naive_estimate(data, k, epsilon)
                data = np.asarray(data)
                gap_err = np.sum(np.square(data[i1] - r1)) / float(len(r1))
                lap_err = np.sum(np.square(data[i2] - r2)) / float(len(r2))
                results_1.append(gap_err)
                results_2.append(lap_err)
            results_1 = np.transpose(results_1)
            results_2 = np.transpose(results_2)

            metric_data[0].append(results_1.mean())
            err_data[0].append(
                [results_1.mean() - results_1.min(), results_1.max() - results_1.mean()])
            metric_data[1].append(results_2.mean())
            err_data[1].append(
                [results_2.mean() - results_2.min(), results_2.max() - results_2.mean()])

        # plot and save
        plt.errorbar(k_array, metric_data[0], yerr=np.transpose(err_data[0]),
                     label='\\huge Gap K Noisy Max', fmt='-o', markersize=12)
        plt.errorbar(k_array, metric_data[1], yerr=np.transpose(err_data[1]),
                     label='\\huge Naive Approach', fmt='-s', markersize=12)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend()
        plt.ylabel('{}'.format('Mean Square Error'), fontsize=24)
        plt.tight_layout()
        plt.savefig('{}/{}.pdf'.format(path_prefix, name).replace(' ', '_'))
        plt.clf()

        logger.info('Figures saved to {}'.format(output_folder))
