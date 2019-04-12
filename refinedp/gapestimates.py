import logging
import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product
import tqdm
from refinedp.algorithms import sparse_vector, noisy_top_k, laplace_mechanism


logger = logging.getLogger(__name__)


def gap_noisy_max(q, epsilon):
    i, imax, max_val, gap = 0, 1, 0, 0
    while i < len(q):
        eta_i = np.random.laplace(scale=2.0 / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i > max_val or i == 1:
            imax = i
            gap = noisy_q_i - max_val
            max_val = noisy_q_i
        else:
            if noisy_q_i > max_val - gap:
                gap = max_val - noisy_q_i
    return imax, gap


def gap_noisy_topk(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(scale=2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps


def gap_topk_estimates_baseline(q, epsilon, k):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices = noisy_top_k(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    return indices, estimates


def gap_topk_estimates(q, epsilon, k):
    indices, gaps = gap_noisy_topk(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    estimates.shape = len(estimates), 1
    gaps.shape = len(gaps), 1
    X = (np.full((k, k), 1) + np.eye(k, k) * 4 * k) * (1.0 / (5 * k))
    Y = np.tile(np.fromiter((k - (i + 1) for i in range(k - 1)), dtype=np.float, count=k - 1), (k, 1))
    Y = Y - np.tri(k, k - 1, -1) * k
    Y = (1.0 / (5 * k)) * Y
    final_estimates = X @ estimates + Y @ gaps
    final_estimates = np.asarray(final_estimates.transpose()[0])
    return indices, final_estimates


def gap_sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(scale=1.0 / epsilon_1)
    while i < len(q) and count < k:
        noisy_q_i = q[i] + np.random.laplace(scale=2.0 * k / epsilon_2)
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    return np.asarray(indices), np.asarray(gaps)


def gap_svt_estimates(q, epsilon, k, threshold):
    # budget allocation for gap svt
    x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    x, y = 1, 1
    gap_budget, lap_budget = y / (x + y), x / (x + y)

    indices, gaps = gap_sparse_vector(q, gap_budget * epsilon, k, threshold, allocation=(gap_x, gap_y))
    assert len(indices) == len(gaps)
    initial_estimates = gaps + threshold
    direct_estimates = laplace_mechanism(q, lap_budget * epsilon, indices)
    variance_gap = 8 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)
    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # variance_lap = 2.0 * c * c / ((epsilon * lap_budget) * (epsilon * lap_budget))
    # variance_gap = (32 + 128 * np.square(c)) / np.square(epsilon)

    # do weighted average
    return indices, (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)


def gap_svt_estimates_baseline(q, epsilon, k, threshold):
    x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)
    indices = sparse_vector(q, epsilon / 2.0, k, threshold, allocation=(gap_x, gap_y))
    return indices, np.asarray(laplace_mechanism(q, epsilon / 2.0, indices))


def mean_square_error(indices, estimates, truth_indices, truth_estimates):
    return np.sum(np.square(truth_estimates - estimates)) / float(len(estimates))


def _evaluate_algorithm(iterations, algorithm, dataset, kwargs, metrics, truth_indices):
    np.random.seed()

    # run several times and record average and error
    results = [[] for _ in range(len(metrics))]
    for _ in range(iterations):
        indices, estimates = algorithm(dataset, **kwargs)
        for metric_index, metric_func in enumerate(metrics):
            results[metric_index].append(metric_func(indices, estimates, truth_indices, dataset[indices]))
    # returns a numpy array of sum of `iterations` runs for each metric
    return np.fromiter((sum(result) for result in results), dtype=np.float, count=len(results))


def evaluate(algorithms, epsilons, input_data,
             metrics=(mean_square_error, ), k_array=np.array(range(2, 25)), total_iterations=20000):
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
        for epsilon, algorithm, k in tqdm.tqdm(product(epsilons, algorithms, k_array),
                                               total=len(epsilons) * len(algorithms) * len(k_array)):
            # for svts
            kwargs = {}
            if 'threshold' in algorithm.__code__.co_varnames:
                threshold = (dataset[sorted_indices[k]] + dataset[sorted_indices[k + 1]]) / 2.0
                kwargs['threshold'] = threshold
            truth_indices = sorted_indices[:k]
            kwargs['epsilon'] = epsilon
            kwargs['k'] = k

            # get the iteration list
            iterations = [int(total_iterations / mp.cpu_count()) for _ in range(mp.cpu_count())]
            iterations[mp.cpu_count() - 1] += total_iterations % mp.cpu_count()

            partial_evaluate_algorithm = \
                partial(_evaluate_algorithm, algorithm=algorithm, dataset=dataset, kwargs=kwargs, metrics=metrics,
                        truth_indices=truth_indices)
            algorithm_metrics = sum(pool.imap_unordered(partial_evaluate_algorithm, iterations)) / total_iterations

            for metric_index, metric in enumerate(metrics):
                metric_data[epsilon][metric.__name__][algorithm.__name__].append(algorithm_metrics[metric_index])

    logger.debug(metric_data)
    return metric_data

