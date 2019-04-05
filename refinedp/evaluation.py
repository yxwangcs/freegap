import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product


logger = logging.getLogger(__name__)


def mean_square_error(indices, estimates, truth_indices, truth_estimates):
    return np.sum(np.square(truth_estimates - estimates)) / float(len(estimates))


def above_threshold_answers(indices, estimates, truth_indices, truth_estimates):
    return len(indices)


def precision(indices, estimates, truth_indices, truth_estimates):
    return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def _evaluate_algorithm(iterations, kwargs, metrics, truth_indices, dataset):
    np.random.seed()
    # run several times and record average and error
    results = [[] for _ in range(len(metrics))]
    for _ in range(iterations):
        indices, estimates = algorithm(dataset, epsilon, k, **kwargs)
        for metric_index, metric_func in enumerate(metrics):
            results[metric_index].append(
                metric_func(indices, estimates, truth_indices, dataset[indices]))

    return np.fromiter((np.asarray(result).mean() for result in results), dtype=np.float, count=len(results))


def evaluate(algorithms, epsilons, input_data, metrics, output_folder='./figures/', k_array=np.array(range(2, 25)),
             algorithm_names=None):
    if algorithm_names is not None:
        assert len(algorithm_names) == len(algorithms), 'algorithm_names must contain names for all algorithms'
    else:
        algorithm_names = tuple(algorithm.__name__.replace('_', ' ').title() for algorithm in algorithms)
    # flatten epsilon
    epsilons = (epsilons, ) if isinstance(epsilons, (int, float)) else epsilons

    # create the output folder if not exists
    output_folder = '{}/{}'.format(os.path.abspath(output_folder), algorithms[0].__name__)
    os.makedirs(output_folder, exist_ok=True)
    output_prefix = os.path.abspath(output_folder)

    # unpack the input data
    dataset_name, dataset = input_data
    dataset = np.asarray(dataset)
    sorted_indices = np.argsort(dataset)[::-1]
    logger.info('Evaluating {} on {}'.format(algorithms[0].__name__.replace('_', ' ').title(), dataset_name))

    total_iterations = 100
    # create the result
    metric_data = {epsilon: [[[] for _ in range(len(algorithms))] for _ in range(len(metrics))] for epsilon in epsilons}
    with mp.Pool(mp.cpu_count()) as pool:
        for epsilon, (algorithm_index, algorithm), k in product(epsilons, enumerate(algorithms), k_array):
            # get the iteration list
            iterations = [int(total_iterations / mp.cpu_count()) for _ in range(mp.cpu_count())]
            iterations[mp.cpu_count() - 1] += total_iterations % mp.cpu_count()

            partial_evaluate_algorithm = \
                partial(_evaluate_algorithm, algorithm=algorithm, epsilon=epsilon, metrics=metrics, dataset=dataset,
                        sorted_indices=sorted_indices)

            # for svts
            kwargs = {}
            threshold_index = 2 * k if 'adaptive' in algorithm.__name__ else k
            if 'threshold' in algorithm.__code__.co_varnames:
                threshold = (dataset[sorted_indices[threshold_index]] + dataset[
                    sorted_indices[threshold_index + 1]]) / 2.0
                kwargs['threshold'] = threshold
            truth_indices = sorted_indices[:threshold_index]

            for local_metric_data in pool.map(partial_evaluate_algorithm, k_array):
                for metric_index in range(len(metrics)):
                    metric_data[epsilon][metric_index][algorithm_index].append(local_metric_data[metric_index])

    logger.info('Figures saved to {}'.format(output_prefix))


def plot(metrics, c_array, metric_data, algorithm_names):
    # plot and save
    formats = ['-o', '-s']
    markers = ['o', 's', '^']
    for metric_index, metric_func in enumerate(metrics):
        metric_name = metric_func.__name__.replace('_', ' ').title()
        for algorithm_index in range(len(algorithm_names)):
            for epsilon_index, epsilon in enumerate(epsilons):
                if metric_func == mean_square_error:
                    if algorithm_index == 0:
                        continue
                    plt.plot(c_array,
                             100 * (np.asarray(metric_data[epsilon][0][0] - np.asarray(
                                 metric_data[epsilon][0][algorithm_index]))) / np.asarray(metric_data[epsilon][0][0]),
                             label='\\huge {}'.format(algorithm_names[algorithm_index]),
                             marker=markers[epsilon_index % len(markers)], linewidth=3,
                             markersize=10)
                    plt.ylim(0, 30)
                    plt.ylabel('\\huge \\% Improvement in MSE')
                else:
                    plt.errorbar(c_array, metric_data[epsilon][metric_index][algorithm_index],
                                 # yerr=np.transpose(err_data[metric_index][algorithm_index]),
                                 label='\\huge {}'.format(algorithm_names[algorithm_index]),
                                 fmt=formats[algorithm_index % len(formats)], linewidth=3, markersize=10)
                    plt.ylabel('{}'.format(metric_name), fontsize=24)
        if metric_func == precision:
            plt.ylim(0.0, 1.0)
        plt.xlabel('\\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        plt.savefig('{}/{}-{}.pdf'.format(output_prefix, dataset_name, metric_name.replace(' ', '_')))
        plt.clf()