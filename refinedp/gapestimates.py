import logging
import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import product
import tqdm
from numba import jit
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


# classical algorithms as building blocks
@jit(nopython=True)
def sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices = []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_1)
    while i < len(q) and count < k:
        if q[i] + np.random.laplace(0, 2.0 * k / epsilon_2) >= noisy_threshold:
            indices.append(i)
            count += 1
        i += 1
    return np.asarray(indices)


def noisy_top_k(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    # counting queries case
    noisy_q = q + np.random.laplace(scale=k / epsilon, size=len(q))
    # otherwise
    # noisy_q = q + np.random.laplace(scale=2 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    return indices


def laplace_mechanism(q, epsilon, indices):
    request_q = q[indices]
    return request_q + np.random.laplace(scale=float(len(request_q)) / epsilon, size=len(request_q))


# implementation of Noisy Max with Measures / Sparse Vector with Measures
# Noisy Top-K with Gap
def gap_noisy_topk(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    # counting queries case
    noisy_q = q + np.random.laplace(scale=k / epsilon, size=len(q))
    # otherwise
    # noisy_q = q + np.random.laplace(scale=2 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps


# Baseline algorithm for Noisy Top-K with Measures
def gap_topk_estimates_baseline(q, epsilon, k):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices = noisy_top_k(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    return indices, estimates


# Noisy Top-K with Measures
def gap_topk_estimates(q, epsilon, k):
    indices, gaps = gap_noisy_topk(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    p_total = (np.fromiter((k - i for i in range(1, k)), dtype=np.int, count=k - 1) * gaps).sum()
    p = np.empty(k, dtype=np.float)
    np.cumsum(gaps, out=p[1:])
    p[0] = 0
    # counting queries case
    final_estimates = (estimates.sum() + k * estimates + p_total - k * p) / (2 * k)
    # otherwise
    #final_estimates = (estimates.sum() + k * estimates + p_total - k * p) / (5 * k)
    return indices, final_estimates


# Sparse Vector with Gap
@jit(nopython=True)
def gap_sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices, gaps = [], []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / epsilon_1)
    while i < len(q) and count < k:
        # counting queries
        noisy_q_i = q[i] + np.random.laplace(0, k / epsilon_2)
        if noisy_q_i >= noisy_threshold:
            indices.append(i)
            gaps.append(noisy_q_i - noisy_threshold)
            count += 1
        i += 1
    return np.asarray(indices), np.asarray(gaps)


# Sparse Vector with Measures
def gap_svt_estimates(q, epsilon, k, threshold):
    # budget allocation for gap svt
    # counting queries
    x, y = 1, np.power(k, 2.0 / 3.0)
    #x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    x, y = 1, 1
    gap_budget, lap_budget = y / (x + y), x / (x + y)

    indices, gaps = gap_sparse_vector(q, gap_budget * epsilon, k, threshold, allocation=(gap_x, gap_y))
    assert len(indices) == len(gaps)
    initial_estimates = gaps + threshold
    direct_estimates = laplace_mechanism(q, lap_budget * epsilon, indices)
    # counting queries
    variance_gap = 8 * np.power((1 + np.power(k, 2.0 / 3)), 3) / np.square(epsilon)
    #variance_gap = 8 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)
    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # do weighted average
    return indices, (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)


# baseline algorithm for Sparse Vector with Measures
def gap_svt_estimates_baseline(q, epsilon, k, threshold):
    x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)
    indices = sparse_vector(q, epsilon / 2.0, k, threshold, allocation=(gap_x, gap_y))
    return indices, np.asarray(laplace_mechanism(q, epsilon / 2.0, indices))


# metric functions
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


def evaluate(algorithms, input_data, epsilons,
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
                threshold = dataset[sorted_indices[int(0.05 * len(sorted_indices))]]
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


def plot(k_array, dataset_name, data, output_prefix, theoretical, algorithm_name, baseline_name):
    generated_files = []
    theoretical_x = np.arange(k_array.min(), k_array.max())
    theoretical_y = theoretical(theoretical_x)
    improves_for_epsilons = []
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict[baseline_name])
        for algorithm, algorithm_data in metric_dict.items():
            if algorithm == baseline_name:
                continue
            improvements = 100 * (baseline - np.asarray(algorithm_data)) / baseline
            improves_for_epsilons.append(improvements[8])
            plt.plot(k_array, improvements, label=r'\huge {}'.format(algorithm_name), linewidth=3, markersize=10,
                     marker='o')
            plt.ylim(0, 50)
            plt.ylabel(r'\huge \% Improvement in MSE')
        plt.plot(theoretical_x, 100 * theoretical_y, linewidth=5,
                 linestyle='--', label=r'\huge Theoretical Expected Improvement')
        plt.xlabel(r'\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend(loc=3)
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        if float(epsilon) - 0.7 < abs(1e-5):
            logger.info('Fix-epsilon Figures saved to {}'.format(output_prefix))
            filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',
                                                 str(epsilon).replace('.', '-'))
            plt.savefig(filename)
            generated_files.append(filename)
        plt.clf()

    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    plt.plot(epsilons, improves_for_epsilons, label=r'\huge {}'.format(algorithm_name), linewidth=3,
             markersize=10, marker='o')
    plt.plot(epsilons, [100 * theoretical(10) for _ in range(len(epsilons))], linewidth=5,
             linestyle='--', label=r'\huge Theoretical Expected Improvement')
    plt.ylabel(r'\huge \% Improvement in MSE')
    plt.ylim(0, 50)
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(np.arange(epsilons.min(), epsilons.max() + 0.1, 0.2))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Fix-k Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-epsilons.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',)
    plt.savefig(filename)
    generated_files.append(filename)
    plt.clf()
    return generated_files
