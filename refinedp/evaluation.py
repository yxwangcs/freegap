import logging
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


def _evaluate_algorithm(iterations, algorithm, dataset, kwargs, metrics, truth_indices):
    np.random.seed()

    # run several times and record average and error
    results = [[] for _ in range(len(metrics))]
    for _ in range(iterations):
        indices, estimates = algorithm(dataset, **kwargs)
        for metric_index, metric_func in enumerate(metrics):
            results[metric_index].append(
                metric_func(indices, estimates, truth_indices, dataset[indices]))
    # returns a numpy array of sum of `iterations` runs for each metric
    return np.fromiter((sum(result) for result in results), dtype=np.float, count=len(results))


def evaluate(algorithms, epsilons, input_data, metrics, k_array=np.array(range(2, 25)), total_iterations=100000):
    # flatten epsilon
    epsilons = (epsilons, ) if isinstance(epsilons, (int, float)) else epsilons

    # unpack the input data
    dataset_name, dataset = input_data
    dataset = np.asarray(dataset)
    sorted_indices = np.argsort(dataset)[::-1]
    logger.info('Evaluating {} on {}'.format(algorithms[-1].__name__.replace('_', ' ').title(), dataset_name))

    # create the result dict
    metric_data = {
        epsilon: {
            metric.__name__: {algorithm.__name__: [] for algorithm in algorithms} for metric in metrics
        } for epsilon in epsilons
    }
    with mp.Pool(mp.cpu_count()) as pool:
        for epsilon, algorithm, k in product(epsilons, algorithms, k_array):
            # get the iteration list
            iterations = [int(total_iterations / mp.cpu_count()) for _ in range(mp.cpu_count())]
            iterations[mp.cpu_count() - 1] += total_iterations % mp.cpu_count()

            # for svts
            kwargs = {}
            threshold_index = 2 * k if 'adaptive' in algorithm.__name__ else k
            if 'threshold' in algorithm.__code__.co_varnames:
                threshold = (dataset[sorted_indices[threshold_index]] +
                             dataset[sorted_indices[threshold_index + 1]]) / 2.0
                kwargs['threshold'] = threshold
            truth_indices = sorted_indices[:threshold_index]
            kwargs['epsilon'] = epsilon
            kwargs['k'] = k

            partial_evaluate_algorithm = \
                partial(_evaluate_algorithm, algorithm=algorithm, dataset=dataset, kwargs=kwargs, metrics=metrics,
                        truth_indices=truth_indices)
            algorithm_metrics = sum(pool.imap(partial_evaluate_algorithm, iterations)) / total_iterations

            for metric_index, metric in enumerate(metrics):
                metric_data[epsilon][metric.__name__][algorithm.__name__].append(algorithm_metrics[metric_index])

    logger.debug(metric_data)
    return metric_data
