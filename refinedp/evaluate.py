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
    baseline_results = [[] for _ in range(len(metrics))]
    algorithm_results = [[] for _ in range(len(metrics))]
    for _ in range(iterations):
        algorithm_result = algorithm(dataset, **kwargs)
        baseline_result = algorithm_result[int(len(algorithm_result) / 2):len(algorithm_result)]
        algorithm_result = algorithm_result[0:int(len(algorithm_result) / 2)]
        indices = algorithm_result[0]
        for metric_index, metric_func in enumerate(metrics):
            baseline_results[metric_index].append(metric_func(*baseline_result, truth_indices, dataset[indices]))
            algorithm_results[metric_index].append(metric_func(*algorithm_result, truth_indices, dataset[indices]))

    # returns a numpy array of sum of `iterations` runs for each metric
    return np.fromiter((sum(result) for result in baseline_results), dtype=np.float, count=len(baseline_results)), \
           np.fromiter((sum(result) for result in algorithm_results), dtype=np.float, count=len(algorithm_results))


def evaluate(algorithm, input_data, epsilons, metrics, k_array=np.array(range(2, 25)), total_iterations=20000,
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
            metric.__name__: {'baseline': [], 'algorithm': []} for metric in metrics
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
            baseline_metrics, algorithm_metrics = np.zeros((len(metrics),)), np.zeros((len(metrics),))
            for local_baseline, local_algorithm in pool.imap_unordered(partial_evaluate_algorithm, iterations):
                baseline_metrics += local_baseline
                algorithm_metrics += local_algorithm
            baseline_metrics = baseline_metrics / total_iterations
            algorithm_metrics = algorithm_metrics / total_iterations

            # merge the results
            for metric_index, metric in enumerate(metrics):
                metric_data[str(epsilon)][metric.__name__]['baseline'].append(baseline_metrics[metric_index])
                metric_data[str(epsilon)][metric.__name__]['algorithm'].append(algorithm_metrics[metric_index])

    logger.debug(metric_data)
    return metric_data
