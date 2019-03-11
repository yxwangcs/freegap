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


def mean_square_error(truth, estimates):
    return 0.0 if estimates is None else np.sum(np.square(truth - estimates)) / float(len(estimates))


def evaluate(algorithms, epsilon, input_data, output_folder='./figures/', c_array=np.array(range(25, 325, 25)),
             metrics=(mean_square_error, ), algorithm_names=None):
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
    logger.info('Evaluating {} on {}'.format(algorithms[0].__name__.replace('_', ' ').title(), dataset_name))

    for metric_func in metrics:
        metric_name = metric_func.__name__.replace('_', ' ').title()

        metric_data = [[] for _ in range(len(algorithms))]
        err_data = [[] for _ in range(len(algorithms))]
        for algorithm_index, algorithm in enumerate(algorithms):
            for c in c_array:
                # for svts
                kwargs = {}
                if 'threshold' in algorithm.__code__.co_varnames:
                    sorted_data = np.sort(dataset)[::-1]
                    threshold = (sorted_data[c] + sorted_data[c + 1]) / 2.0
                    kwargs['threshold'] = threshold

                results = []
                # run several times and record average and error
                for _ in range(10):
                    indices, estimates = algorithm(dataset, epsilon, c, **kwargs)
                    results.append(metric_func(dataset[indices], estimates))
                results = np.asarray(results)

                metric_data[algorithm_index].append(results.mean())
                err_data[algorithm_index].append([results.mean() - results.min(), results.max() - results.mean()])

        # plot and save
        formats = ['-o', '-s']
        for algorithm_index in range(len(algorithms)):
            plt.errorbar(c_array, metric_data[algorithm_index], yerr=np.transpose(err_data[algorithm_index]),
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
