import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from functools import partial


logger = logging.getLogger(__name__)


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


def precision(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    return len(np.intersect1d(indices, truth_indices)) / float(len(indices))


def recall(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    return len(np.intersect1d(indices, truth_indices)) / float(2 * c_val)


def f_measure(sorted_indices, c_val, indices, truth_indices, truth_estimates, estimates):
    precision = len(np.intersect1d(indices, truth_indices)) / float(len(indices))
    recall = len(np.intersect1d(indices, truth_indices)) / float(2 * c_val)
    return 2 * precision * recall / (precision + recall)


def evaluate_c(c, algorithm, epsilon, metrics, dataset, sorted_indices):
    np.random.seed()
    # for svts
    kwargs = {}
    threshold_index = 2 * c if 'adaptive' in algorithm.__name__ else c
    if 'threshold' in algorithm.__code__.co_varnames:
        threshold = (dataset[sorted_indices[threshold_index]] + dataset[sorted_indices[threshold_index + 1]]) / 2.0
        kwargs['threshold'] = threshold

    truth_indices = sorted_indices[:threshold_index]

    # run several times and record average and error
    results = [[] for _ in range(len(metrics))]
    for _ in range(2000):
        indices, estimates = algorithm(dataset, epsilon, c, **kwargs)
        for metric_index, metric_func in enumerate(metrics):
            results[metric_index].append(
                metric_func(sorted_indices, c, indices, truth_indices, dataset[indices], estimates))

    results = np.asarray(results)
    final_metric_data = []
    final_error_data = []
    for metric_index in range(len(metrics)):
        final_metric_data.append(results[metric_index].mean())
        final_error_data.append((results[metric_index].mean() - results[metric_index].min(),
                                 results[metric_index].max() - results[metric_index].mean()))

    return final_metric_data, final_error_data


def evaluate(algorithms, epsilons, input_data, output_folder='./figures/', c_array=np.array(range(2, 25)),
             metrics=(mean_square_error, above_threshold_answers, accuracy, normalized_cumulative_rank),
             algorithm_names=None):
    if algorithm_names is not None:
        assert len(algorithm_names) == len(algorithms), 'algorithm_names must contain names for all algorithms'
    else:
        algorithm_names = [algorithm.__name__.replace('_', ' ').title() for algorithm in algorithms]
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

    with mp.Pool(mp.cpu_count()) as pool:
        metric_data = {}
        err_data = {}
        for epsilon in epsilons:
            metric_data[epsilon] = [[[] for _ in range(len(algorithms))] for _ in range(len(metrics))]
            err_data[epsilon] = [[[] for _ in range(len(algorithms))] for _ in range(len(metrics))]
            for algorithm_index, algorithm in enumerate(algorithms):
                partial_evaluate_c = partial(evaluate_c,
                                             algorithm=algorithm, epsilon=epsilon, metrics=metrics, dataset=dataset,
                                             sorted_indices=sorted_indices)
                for local_metric_data, local_error_data in pool.map(partial_evaluate_c, c_array):
                    for metric_index in range(len(metrics)):
                        metric_data[epsilon][metric_index][algorithm_index].append(local_metric_data[metric_index])
                        err_data[epsilon][metric_index][algorithm_index].append(local_error_data[metric_index])
    # plot and save
    formats = ['-o', '-s']
    markers = ['o', 's', '^']
    for metric_index, metric_func in enumerate(metrics):
        metric_name = metric_func.__name__.replace('_', ' ').title()
        for algorithm_index in range(len(algorithms)):
            for epsilon_index, epsilon in enumerate(epsilons):
                if metric_func == mean_square_error:
                    if algorithm_index == 0:
                        continue
                    plt.plot(c_array,
                             100 * (np.asarray(metric_data[epsilon][0][0] - np.asarray(metric_data[epsilon][0][algorithm_index]))) / np.asarray(metric_data[epsilon][0][0]),
                             label='\\huge {}'.format(algorithm_names[algorithm_index]),
                             marker=markers[epsilon_index % len(markers)], linewidth=3,
                             markersize=10)
                    plt.ylim(0, 30)
                    plt.ylabel('\\huge Mean Square Error Improvement \\%')
                else:
                    plt.errorbar(c_array, metric_data[epsilon][metric_index][algorithm_index],
                                 # yerr=np.transpose(err_data[metric_index][algorithm_index]),
                                 label='\\huge {}'.format(algorithm_names[algorithm_index]),
                                 fmt=formats[algorithm_index % len(formats)], linewidth=3, markersize=10)
                    plt.ylabel('{}'.format(metric_name), fontsize=24)
        if metric_func == precision:
            plt.ylim(0.0, 1.0)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        plt.savefig('{}/{}-{}.pdf'.format(output_prefix, dataset_name, metric_name.replace(' ', '_')))
        plt.clf()

    logger.info('Figures saved to {}'.format(output_prefix))
