import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product
import logging
import tqdm

logger = logging.getLogger(__name__)


def _evaluate_algorithm(iterations, algorithm, dataset, kwargs, metrics, truth_indices):
    np.random.seed()

    # run several times and record average and error
    all_results = []
    for _ in range(iterations):
        results = algorithm(dataset, **kwargs)
        # initialize results list
        if len(all_results) != len(results):
            for _ in range(len(results)):
                all_results.append([[] for _ in range(len(metrics))])

        # add result into all_results list, the last one indicates the baseline result
        for i, result in enumerate(results):
            indices = result[0]
            for metric_index, metric_func in enumerate(metrics):
                all_results[i][metric_index].append(metric_func(*result, truth_indices, dataset[indices]))

    # returns a numpy array of sum of `iterations` runs for each metric
    return tuple(
        np.fromiter((sum(result) for result in algorithm_result), dtype=np.float64, count=len(algorithm_result))
        for algorithm_result in all_results
    )


def evaluate(algorithm, input_data, epsilons, metrics, k_array=np.array(range(2, 25)), total_iterations=10000,
             counting_queries=False):
    # make epsilons a tuple
    epsilons = (epsilons, ) if isinstance(epsilons, (int, float)) else tuple(epsilons)

    # unpack the input data
    dataset_name, dataset = input_data
    dataset = np.asarray(dataset)
    sorted_indices = np.argsort(dataset)[::-1]
    logger.info('Evaluating {} on {}'.format(algorithm.__name__.replace('_', ' ').title(), dataset_name))

    # create the result dict
    metric_data = {
        str(epsilon): {
            metric.__name__: [] for metric in metrics
        } for epsilon in epsilons
    }
    with mp.Pool(mp.cpu_count()) as pool:
        for epsilon, k in tqdm.tqdm(product(epsilons, k_array), total=len(epsilons) * len(k_array)):
            # for svts
            kwargs = {}
            if 'threshold' in algorithm.__code__.co_varnames:
                if 'adaptive' in algorithm.__name__:
                    kwargs['threshold'] = dataset[sorted_indices[int(0.05 * len(sorted_indices))]]
                    truth_indices = sorted_indices[:int(0.05 * len(sorted_indices))]
                else:
                    #kwargs['threshold'] = (dataset[sorted_indices[k]] + dataset[sorted_indices[k + 1]]) / 2.0
                    kwargs['threshold'] = dataset[sorted_indices[50]]
                    truth_indices = sorted_indices[:50]
            else:
                truth_indices = sorted_indices[:k]
            kwargs['epsilon'] = epsilon
            kwargs['k'] = k
            kwargs['counting_queries'] = counting_queries

            # get the iteration list
            iterations = [int(total_iterations / mp.cpu_count()) for _ in range(mp.cpu_count())]
            iterations[mp.cpu_count() - 1] += total_iterations % mp.cpu_count()

            partial_evaluate_algorithm = \
                partial(_evaluate_algorithm, algorithm=algorithm, dataset=dataset, kwargs=kwargs, metrics=metrics,
                        truth_indices=truth_indices)

            # run and collect data
            metric_results = []
            for local_result in pool.imap_unordered(partial_evaluate_algorithm, iterations):
                # initialize metric_results
                if len(metric_results) != len(local_result):
                    for _ in range(len(local_result)):
                        metric_results.append(np.zeros((len(metrics),)))

                for i, result in enumerate(local_result):
                    metric_results[i] += result

            for i in range(len(metric_results)):
                metric_results[i] = metric_results[i] / total_iterations

            # merge the results
            for i, metric_result in enumerate(metric_results):
                for metric_index, metric in enumerate(metrics):
                    # initialize metric_data list
                    if len(metric_data[str(epsilon)][metric.__name__]) != len(metric_results):
                        for _ in range(len(metric_results)):
                            metric_data[str(epsilon)][metric.__name__].append([])
                    # write data
                    metric_data[str(epsilon)][metric.__name__][i].append(metric_result[metric_index])

    logger.debug(metric_data)
    return metric_data
