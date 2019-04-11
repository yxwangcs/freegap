import logging
import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product
import tqdm


logger = logging.getLogger(__name__)


def adaptive_sparse_vector(q, epsilon, k, threshold, top_prng=np.random, middle_prng=np.random):
    indices, top_indices, middle_indices = [], [], []
    epsilon_0, epsilon_1, epsilon_2 = epsilon / 2.0, epsilon / (8.0 * k), epsilon / (4.0 * k)
    sigma = 2 * np.sqrt(2) / epsilon_1
    i, priv = 0, epsilon_0
    noisy_threshold = threshold + top_prng.laplace(scale=1.0 / epsilon_0)
    while i < len(q) and priv <= epsilon - 2 * epsilon_2:
        eta_i = top_prng.laplace(scale=1.0 / epsilon_1)
        xi_i = middle_prng.laplace(scale=1.0 / epsilon_2)
        if q[i] + eta_i - noisy_threshold >= sigma:
            indices.append(i)
            top_indices.append(i)
            priv += 2 * epsilon_1
        elif q[i] + xi_i - noisy_threshold >= 0:
            indices.append(i)
            middle_indices.append(i)
            priv += 2 * epsilon_2
        i += 1
    logger.debug('Total refined: {}'.format(len(top_indices)))

    return np.asarray(indices), np.asarray(top_indices), np.asarray(middle_indices)


def sparse_vector(q, epsilon, k, threshold, middle_prng=np.random):
    indices = []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(scale=2.0 / epsilon)
    while i < len(q) and count < k:
        if q[i] + middle_prng.laplace(scale=4.0 * k / epsilon) >= noisy_threshold:
            indices.append(i)
            count += 1
        i += 1
    return np.asarray(indices), np.asarray(indices), np.asarray([])


def above_threshold_answers(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(indices)


def precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def recall(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(np.intersect1d(indices, truth_indices)) / float(len(truth_indices))


def top_branch(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(top_indices)


def middle_branch(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(middle_indices)


def top_branch_precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    if len(top_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(top_indices, truth_indices)) / float(len(top_indices))


def top_branch_recall(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(np.intersect1d(top_indices, truth_indices)) / float(len(truth_indices))


def middle_branch_precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    if len(middle_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(middle_indices, truth_indices)) / float(len(middle_indices))


def middle_branch_recall(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(np.intersect1d(middle_indices, truth_indices)) / float(len(truth_indices))


def left_epsilon(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    baseline_indices, *_ = baseline_result
    stopped_index = baseline_indices.max()
    left_privacy = np.count_nonzero(top_indices > stopped_index) * 0.25 / k + \
                   np.count_nonzero(middle_indices > stopped_index) * 0.5 / k
    return left_privacy


def _evaluate_algorithm(iterations, algorithms, dataset, kwargs, metrics, truth_indices):
    np.random.seed()
    baseline, algorithm = algorithms

    # run several times and record average and error
    baseline_results = [[] for _ in range(len(metrics))]
    algorithm_results = [[] for _ in range(len(metrics))]
    for _ in range(iterations):
        baseline_result = baseline(dataset, **kwargs)
        algorithm_result = algorithm(dataset, **kwargs)
        for metric_index, metric_func in enumerate(metrics):
            baseline_results[metric_index].append(metric_func(*baseline_result, baseline_result, truth_indices, kwargs['k']))
            algorithm_results[metric_index].append(metric_func(*algorithm_result, baseline_result, truth_indices, kwargs['k']))

    # returns a numpy array of sum of `iterations` runs for each metric
    return np.fromiter((sum(result) for result in baseline_results), dtype=np.float, count=len(baseline_results)),\
           np.fromiter((sum(result) for result in algorithm_results), dtype=np.float, count=len(algorithm_results))


def evaluate(algorithms, epsilons, input_data,
             metrics=(above_threshold_answers, precision, top_branch, middle_branch, top_branch_precision,
                      middle_branch_precision, left_epsilon),
             k_array=np.array(range(2, 25)), total_iterations=20000):
    assert len(algorithms) == 2, 'algorithms must contain baseline and the algorithm to evaluate'
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
        for epsilon, k in tqdm.tqdm(product(epsilons, k_array), total=len(epsilons) * len(k_array)):
            # get the iteration list
            iterations = [int(total_iterations / mp.cpu_count()) for _ in range(mp.cpu_count())]
            iterations[mp.cpu_count() - 1] += total_iterations % mp.cpu_count()

            kwargs = {
                'threshold': (dataset[sorted_indices[2 * k]] + dataset[sorted_indices[2 * k + 1]]) / 2.0,
                'epsilon': epsilon,
                'k': k
            }
            truth_indices = sorted_indices[:2 * k + 1]

            partial_evaluate_algorithm = \
                partial(_evaluate_algorithm, algorithms=algorithms, dataset=dataset, kwargs=kwargs, metrics=metrics,
                        truth_indices=truth_indices)

            baseline_metrics, algorithm_metrics = np.zeros((len(metrics), )), np.zeros((len(metrics), ))
            for local_baseline, local_algorithm in pool.imap_unordered(partial_evaluate_algorithm, iterations):
                baseline_metrics += local_baseline
                algorithm_metrics += local_algorithm
            baseline_metrics, algorithm_metrics = baseline_metrics / total_iterations, algorithm_metrics / total_iterations

            for metric_index, metric in enumerate(metrics):
                metric_data[epsilon][metric.__name__][algorithms[0].__name__].append(baseline_metrics[metric_index])
                metric_data[epsilon][metric.__name__][algorithms[1].__name__].append(algorithm_metrics[metric_index])

    logger.debug(metric_data)
    return metric_data

