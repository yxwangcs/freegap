import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from refinedp.preprocess import process_bms_pos, process_kosarak

__all__ = ['process_datasets', 'evaluate']

logger = logging.getLogger(__name__)


def process_datasets(folder):
    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(folder)
    # yield different datasets with their names
    yield 'BMS-POS', process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder))
    yield 'kosarak', process_kosarak('{}/kosarak.dat'.format(dataset_folder))


def mean_square_error(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    return 0.0 if estimates is None else np.sum(np.square(truth_estimates - estimates)) / float(len(estimates))


def above_threshold_answers(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    return len(indices)


def accuracy(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    total = 0
    for index in indices:
        total += 1 if index in truth_indices else 0
    return float(total) / len(indices)


def normalized_cumulative_rank(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    scores = {}
    for order_index, data_index in enumerate(sorted_indices[:2 * c_val]):
        scores[data_index] = (2 * c_val - order_index)

    total_score = 0
    for index in indices:
        total_score += 0 if index not in scores else scores[index]

    return float(total_score) / (c_val * (2 * c_val + 1))


def evaluate(algorithms, epsilon, input_data, output_folder='./figures/', c_array=np.array(range(25, 325, 25)),
             metrics=(mean_square_error, above_threshold_answers, accuracy, normalized_cumulative_rank),
             algorithm_names=None):
    if algorithm_names is not None:
        assert len(algorithm_names) == len(algorithms), 'algorithm_names must contain names for all algorithms'
    else:
        algorithm_names = [algorithm.__name__.replace('_', ' ').title() for algorithm in algorithms]

    # create the output folder if not exists
    output_folder = '{}/{}'.format(os.path.abspath(output_folder), algorithms[0].__name__)
    os.makedirs(output_folder, exist_ok=True)
    output_prefix = os.path.abspath(output_folder)

    # unpack the input data
    dataset_name, dataset = input_data
    dataset = np.asarray(dataset)
    sorted_indices = np.argsort(dataset)[::-1]
    logger.info('Evaluating {} on {}'.format(algorithms[0].__name__.replace('_', ' ').title(), dataset_name))

    metric_data = [[[] for _ in range(len(algorithms))] for _ in range(len(metrics))]
    err_data = [[[] for _ in range(len(algorithms))] for _ in range(len(metrics))]
    for algorithm_index, algorithm in enumerate(algorithms):
        for c in c_array:
            # for svts
            kwargs = {}
            threshold_index = 2 * c if 'adaptive' in algorithm.__name__ else c
            if 'threshold' in algorithm.__code__.co_varnames:
                threshold = (dataset[sorted_indices[threshold_index]] + dataset[sorted_indices[threshold_index + 1]]) / 2.0
                kwargs['threshold'] = threshold

            truth_indices = sorted_indices[:threshold_index]

            # run several times and record average and error
            results = [[] for _ in range(len(metrics))]
            for _ in range(10):
                indices, estimates = algorithm(dataset, epsilon, c, **kwargs)
                for metric_index, metric_func in enumerate(metrics):
                    results[metric_index].append(
                        metric_func(sorted_indices, c, indices, truth_indices, dataset[indices], estimates))

            results = np.asarray(results)

            for metric_index in range(len(metrics)):
                metric_data[metric_index][algorithm_index].append(results[metric_index].mean())
                err_data[metric_index][algorithm_index].append(
                    (results[metric_index].mean() - results[metric_index].min(),
                     results[metric_index].max() - results[metric_index].mean()))

    # plot and save
    formats = ['-o', '-s']
    for metric_index, metric_func in enumerate(metrics):
        metric_name = metric_func.__name__.replace('_', ' ').title()
        for algorithm_index in range(len(algorithms)):
            plt.errorbar(c_array, metric_data[metric_index][algorithm_index],
                         yerr=np.transpose(err_data[metric_index][algorithm_index]),
                         label='\\huge {}'.format(algorithm_names[algorithm_index]),
                         fmt=formats[algorithm_index % len(formats)], markersize=12)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend()
        plt.ylabel('{}'.format(metric_name), fontsize=24)
        plt.tight_layout()
        plt.savefig('{}/{}-{}.pdf'.format(output_prefix, dataset_name, metric_name))
        plt.clf()

    logger.info('Figures saved to {}'.format(output_prefix))
