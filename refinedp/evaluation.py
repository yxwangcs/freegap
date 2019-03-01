import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from refinedp.preprocess import process_bms_pos, process_kosarak


logger = logging.getLogger(__name__)


def evaluate(test_algorithm, reference_algorithms, epsilon,
             kwargs=None, c_array=np.array(range(25, 325, 25)), dataset_folder='datasets/', output_folder='./figures/'):
    logger.info('Evaluating {}'.format(test_algorithm.__name__))

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

            results_1, results_2, results_3 = [], [], []
            for _ in range(10):
                i1, r1 = refined_estimate_sparse_vector(data, threshold, c, epsilon)
                i2, r2 = naive_estimate_sparse_vector(data, threshold, c, epsilon)
                # i3, r3 = numerical_estimate_sparse_vector(data, threshold, c, epsilon)
                data = np.asarray(data)
                gap_err = np.sum(np.square(data[i1] - r1)) / float(len(r1))
                lap_err = np.sum(np.square(data[i2] - r2)) / float(len(r2))
                # num_err = np.sqrt(np.sum(np.square(data[i3] - r3)) / float(len(r3)))
                if show:
                    print(data[i1], r1)
                    show = False
                results_1.append(gap_err)
                results_2.append(lap_err)
                # results_3.append(num_err)
            results_1 = np.transpose(results_1)
            results_2 = np.transpose(results_2)
            # results_3 = np.transpose(results_3)

            metric_data[0].append(results_1.mean())
            err_data[0].append(
                [results_1.mean() - results_1.min(), results_1.max() - results_1.mean()])
            metric_data[1].append(results_2.mean())
            err_data[1].append(
                [results_2.mean() - results_2.min(), results_2.max() - results_2.mean()])
            # metric_data[2].append(results_3.mean())
            # err_data[2].append(
            # [results_3.mean() - results_3.min(), results_3.max() - results_3.mean()])

        # plot and save
        plt.errorbar(c_array, metric_data[0], yerr=np.transpose(err_data[0]),
                     label='\\huge {}'.format(test_algorithm.__name__), fmt='-o', markersize=12)
        plt.errorbar(c_array, metric_data[1], yerr=np.transpose(err_data[1]),
                     label='\\huge {}'.format(reference_algorithms[0].__name__), fmt='-s', markersize=12)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend()
        plt.ylabel('{}'.format('Mean Square Error'), fontsize=24)
        plt.tight_layout()
        plt.savefig('{}/{}.pdf'.format(path_prefix, name).replace(' ', '_'))
        plt.clf()

        logger.info('Figures saved to {}'.format(output_folder))
