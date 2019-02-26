import logging
import os
import numpy as np
import matplotlib.pyplot as plt
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


def refined_estimate_sparse_vector(q, threshold, c, epsilon):
    answers = np.asarray(gap_sparse_vector(q, threshold, c, epsilon / 2.0))
    indices = np.nonzero(answers)
    initial_estimates = answers[indices] + threshold
    direct_estimates = laplace_mechanism(q, answers, epsilon / 2.0)
    return indices, (8 * c * c * initial_estimates + (32 + 128 * c * c) * direct_estimates) / (8 * c * c + 32 + 128 * c * c)


def naive_estimate_sparse_vector(q, threshold, c, epsilon):
    answers = sparse_vector(q, threshold, c, epsilon / 2.0)
    return np.nonzero(answers), np.asarray(laplace_mechanism(q, answers, epsilon / 2.0))


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

    show = True

    for name, data in datasets.items():
        logger.info('Evaluating on {}'.format(name))
        sorted_data = np.sort(data)[::-1]
        metric_data, err_data = [[], []], [[], []]
        for c in c_array:
            threshold = (sorted_data[c] + sorted_data[c + 1]) / 2.0

            results_1, results_2 = [], []
            for _ in range(10):
                i1, r1 = refined_estimate_sparse_vector(data, threshold, c, epsilon)
                i2, r2 = naive_estimate_sparse_vector(data, threshold, c, epsilon)
                data = np.asarray(data)
                gap_err = np.sqrt(np.sum(np.square(data[i1] - r1)) / float(len(r1)))
                lap_err = np.sqrt(np.sum(np.square(data[i2] - r2)) / float(len(r2)))
                if show:
                    print(data[i1], r1)
                    show = False
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
        plt.errorbar(c_array, metric_data[0], yerr=np.transpose(err_data[0]),
                     label='\\huge Gap Sparse Vector', fmt='-o', markersize=12)
        plt.errorbar(c_array, metric_data[1], yerr=np.transpose(err_data[1]),
                     label='\\huge Naive Sparse Vector', fmt='-s', markersize=12)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        #plt.ylim(0.0, 1.0)
        plt.legend()
        plt.ylabel('{}'.format('Mean Square Error'), fontsize=24)
        plt.tight_layout()
        plt.savefig('{}/{}.pdf'.format(path_prefix, name).replace(' ', '_'))
        plt.clf()

        logger.info('Figures saved to {}'.format(output_folder))



