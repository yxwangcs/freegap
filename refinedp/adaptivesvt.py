import logging
import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product
import tqdm


logger = logging.getLogger(__name__)


def adaptive_sparse_vector(q, epsilon, k, threshold):
    out = []
    top_branch = []
    middle_branch = []
    count = 1
    refine_count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < 2 * k - 1:
        eta_i = np.random.laplace(scale=8.0 * k / epsilon)
        xi_i = np.random.laplace(scale=4.0 * k / epsilon)
        if q[i] + eta_i >= noisy_threshold + 16.0 * np.sqrt(2) * k / epsilon:
            out.append(True)
            top_branch.append(True)
            count += 1
        elif q[i] + xi_i > noisy_threshold:
            refine_count += 1
            out.append(True)
            middle_branch.append(True)
            count += 2
        else:
            out.append(False)
            top_branch.append(False)
            middle_branch.append(False)
        i += 1
    logger.debug('Total refined: {}'.format(refine_count))

    return np.nonzero(out)[0], np.nonzero(top_branch)[0], np.nonzero(middle_branch)[0]


def sparse_vector(q, epsilon, k, threshold):
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < k:
        eta_i = np.random.laplace(scale=4.0 * k / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(True)
            count += 1
        else:
            out.append(False)
        i += 1
    return np.nonzero(out)[0], np.nonzero(out)[0], np.asarray([])


def above_threshold_answers(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(indices)


def precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def top_branch(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(top_indices)


def middle_branch(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    return len(middle_indices)


def top_branch_precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    if len(top_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(top_indices, truth_indices)) / float(len(top_indices))


def middle_branch_precision(indices, top_indices, middle_indices, baseline_result, truth_indices, k):
    if len(middle_indices) == 0:
        return 1.0
    else:
        return len(np.intersect1d(middle_indices, truth_indices)) / float(len(middle_indices))


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
             metrics=(above_threshold_answers, top_branch, middle_branch, top_branch_precision, middle_branch_precision,
                      left_epsilon),
             k_array=np.array(range(2, 25)), total_iterations=100000):
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

