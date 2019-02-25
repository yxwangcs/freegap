import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from refinedp.preprocess import process_kosarak, process_bms_pos
from refinedp.refinelaplace import refinelaplace
from refinedp.algorithms import sparse_vector


logger = logging.getLogger(__name__)


def adaptive_sparse_vector(q, threshold, c, epsilon):
    out = []
    count = 1
    refine_count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < 2 * c - 1:
        eta_i = np.random.laplace(scale=8.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold + 24.0 * np.sqrt(2) * c / epsilon:
            out.append(True)
            count += 1
        else:
            psi_i = refinelaplace(eta_i, epsilon / (4.0 * c), epsilon / (8.0 * c))
            refine_count += 1
            noisy_q_i = q[i] + psi_i
            if noisy_q_i >= noisy_threshold:
                out.append(True)
                count += 2
            else:
                out.append(False)
        i += 1
    logger.debug('Total refined: {}'.format(refine_count))

    return out


def evaluate_adaptive_sparse_vector(dataset_folder='datasets', output_folder='./figures/adaptive-sparse-vector'):
    # create the output folder if not exists
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    path_prefix = os.path.abspath(output_folder)

    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(dataset_folder)
    datasets = {
        'BMS-POS': process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder)),
        'kosarak': process_kosarak('{}/kosarak.dat'.format(dataset_folder))
    }

    c_array = list(range(25, 325, 25))

    METRICS = ['Above-Threshold Answers', 'False Negative Rate']
    # metric_data[x][y] - the x-th metrics on varying c values for y-th algorithm, where y in (0, 1),
    # 0 is adaptive svt and 1 is vanilla svt
    # similar for err_data
    metric_data = [[[], []] for _ in range(len(METRICS))]
    err_data = [[[], []] for _ in range(len(METRICS))]

    def calc_metrics(data, answer, truth):
        # Above-Threshold Answers, false_negative_rate
        return np.count_nonzero(answer), \
               float(np.count_nonzero(answer == (truth[:len(answer)] == False))) / np.count_nonzero(
                   truth[:len(answer)] == False)

    for name, data in datasets.items():
        logger.info('Evaluating on {}'.format(name))
        # plot the histogram of the data
        plt.plot(np.asarray(list(range(len(data)))), data)
        plt.title(name)
        plt.savefig('{}/{}.pdf'.format(path_prefix, name))
        plt.clf()

        # run and gather data
        epsilon = 0.3
        for c in c_array:
            sorted_data = np.sort(data)[::-1]
            threshold = (sorted_data[2 * c] + sorted_data[2 * c + 1]) / 2.0
            truth_values = data > threshold

            results_1, results_2 = [], []
            for _ in range(10):
                r1 = np.asarray(adaptive_sparse_vector(data, threshold, c, epsilon), dtype=np.bool)
                r2 = np.asarray(sparse_vector(data, threshold, c, epsilon), dtype=np.bool)
                results_1.append(calc_metrics(data, r1, truth_values))
                results_2.append(calc_metrics(data, r2, truth_values))
            results_1 = np.transpose(results_1)
            results_2 = np.transpose(results_2)
            for i in range(len(METRICS)):
                metric_data[i][0].append(results_1[i].mean())
                err_data[i][0].append(
                    [results_1[i].mean() - results_1[i].min(), results_1[i].max() - results_1[i].mean()])
                metric_data[i][1].append(results_2[i].mean())
                err_data[i][1].append(
                    [results_2[i].mean() - results_2[i].min(), results_2[i].max() - results_2[i].mean()])

        # plot and save
        for index, metric in enumerate(METRICS):
            plt.errorbar(c_array, metric_data[index][0], yerr=np.transpose(err_data[index][0]),
                         label='\\huge Adaptive Sparse Vector', fmt='-o', markersize=8)
            plt.errorbar(c_array, metric_data[index][1], yerr=np.transpose(err_data[index][1]),
                         label='\\huge Sparse Vector', fmt='-s', markersize=8)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend()
            plt.ylabel('{}'.format(metric), fontsize=18)
            plt.savefig('{}/{}_{}.pdf'.format(path_prefix, name, metric).replace(' ', '_'))
            plt.clf()

    logger.info('Figures saved to {}'.format(output_folder))
