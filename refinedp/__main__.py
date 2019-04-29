import argparse
import os
import difflib
import logging
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import coloredlogs
from refinedp.adaptivesvt import adaptive_sparse_vector, sparse_vector
from refinedp.adaptivesvt import evaluate as evaluate_adaptivesvt
from refinedp.gapestimates import gap_svt_estimates, gap_svt_estimates_baseline, \
    gap_topk_estimates, gap_topk_estimates_baseline
from refinedp.gapestimates import evaluate as evaluate_gap_estimates
from refinedp.preprocess import process_t40100k, process_bms_pos, process_kosarak, process_sf1


matplotlib.use('PDF')


# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},' \
                                             '\\usepackage[libertine]{newtxmath},' \
                                             '\\usepackage{sfmath},' \
                                             '\\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def compress_pdf(file):
    import subprocess
    import os
    os.rename(file, '{}.temp'.format(file))
    subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                     '-dPDFSETTINGS=/default',
                     '-dNOPAUSE', '-dQUIET', '-dBATCH',
                     '-sOutputFile={}'.format(file),
                     '{}.temp'.format(file)]
                    )
    os.remove('{}.temp'.format(file))


def process_datasets(folder):
    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(folder)
    # yield different datasets with their names
    yield 'T40100K', process_t40100k('{}/T40I10D100K.dat'.format(dataset_folder))
    yield 'BMS-POS', process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder))
    yield 'kosarak', process_kosarak('{}/kosarak.dat'.format(dataset_folder))


def plot_adaptive(k_array, dataset_name, data, output_prefix):
    with open('{}/{}.json'.format(output_prefix, dataset_name), 'w') as f:
        json.dump(data, f)

    epsilon = '0.3'
    # first find the best quantile
    """
    quantiles = tuple(data[epsilon]['top_branch']['adaptive_sparse_vector'].keys())
    quantile_scores = []
    for quantile in quantiles:
        top = np.asarray(data[epsilon]['top_branch']['adaptive_sparse_vector'][quantile], dtype=np.int)
        middle = np.asarray(data[epsilon]['middle_branch']['adaptive_sparse_vector'][quantile], dtype=np.int)
        quantile_scores.append((top + middle).sum())
    quantile_scores = np.asarray(quantile_scores)
    quantile = quantiles[quantile_scores.argmax()]
    """
    quantile = '0.05'

    left_epsilons = []
    for _, epsilon_dict in data.items():
        left_epsilons.append(epsilon_dict['left_epsilon']['adaptive_sparse_vector'][quantile][8])
    logger.info('best quantile is {}'.format(quantile))

    # plot number of above threshold answers
    baseline_top_branch = data[epsilon]['top_branch']['sparse_vector'][quantile]
    algorithm_top_branch = data[epsilon]['top_branch']['adaptive_sparse_vector'][quantile]
    algorithm_middle_branch = data[epsilon]['middle_branch']['adaptive_sparse_vector'][quantile]
    algorithm_total = data[epsilon]['above_threshold_answers']['adaptive_sparse_vector'][quantile]
    plt.plot(k_array, baseline_top_branch,
             label='\\huge {}'.format('Classical Sparse Vector'),
             linewidth=3, markersize=10, marker='o')
    plt.plot(k_array, algorithm_total,
             label='\\huge {}'.format('Adaptive SVT w/ Gap (Total)'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, algorithm_top_branch,
             label='\\huge {}'.format('Adaptive SVT w/ Gap (Top)'),
             linewidth=3, markersize=10, marker='s')
    plt.plot(k_array, algorithm_middle_branch,
             label='\\huge {}'.format('Adaptive SVT w/ Gap (Middle)'),
             linewidth=3, markersize=10, marker='^')
    plt.ylim(0, 50)
    plt.ylabel('\\huge {}'.format('\\# of Above-Threshold Answers'))
    plt.xlabel('\\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'above_threshold_answers',
                                         str(epsilon).replace('.', '-')))
    plt.clf()

    # plot the precision
    #baseline_top_branch = data[epsilon]['precision']['sparse_vector']
    #algorithm_top_branch = data[epsilon]['top_branch_precision']['adaptive_sparse_vector']
    #algorithm_middle_branch = data[epsilon]['middle_branch_precision']['adaptive_sparse_vector']
    algorithm_total = data[epsilon]['precision']['adaptive_sparse_vector'][quantile]
    sparse_vector_precision = data[epsilon]['precision']['sparse_vector'][quantile]
    sparse_vector_recall = data[epsilon]['recall']['sparse_vector'][quantile]
    algorithm_recall = data[epsilon]['recall']['adaptive_sparse_vector'][quantile]
    #plt.plot(k_array, baseline_top_branch,
             #label='\\huge {}'.format('Classical Sparse Vector'),
             #linewidth=3, markersize=10, marker='o')
    plt.plot(k_array, sparse_vector_precision,
             label='\\huge {}'.format('Precision - Sparse Vector'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, algorithm_total,
             label='\\huge {}'.format('Precision - Adaptive SVT w/ Gap'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    #plt.plot(k_array, sparse_vector_recall,
             #label='\\huge {}'.format('Recall - Sparse Vector'),
             #linewidth=3, markersize=10, marker='P', zorder=5)
    #plt.plot(k_array, algorithm_recall,
             #label='\\huge {}'.format('Recall - Adaptive SVT w/ Gap'),
             #linewidth=3, markersize=10, marker='P', zorder=5)
    #plt.plot(k_array, algorithm_top_branch,
             #label='\\huge {}'.format('Adaptive SVT w/ Gap (Top)'),
             #linewidth=3, markersize=10, marker='s')
    #plt.plot(k_array, algorithm_middle_branch,
             #label='\\huge {}'.format('Adaptive SVT w/ Gap (Middle)'),
             #linewidth=3, markersize=10, marker='^')
    plt.ylim(0, 1.0)
    plt.ylabel('\\huge {}'.format('Precision and Recall'))
    plt.xlabel('\\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'precision',
                                         str(epsilon).replace('.', '-')))
    plt.clf()

    # plot left epsilons
    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    left_budget = np.asarray(left_epsilons) * 100
    plt.plot(epsilons, left_budget,
             label='\\huge {}'.format('Adaptive Sparse Vector with Gap'),
             linewidth=3, markersize=10, marker='o')
    plt.ylim(0, 25)
    plt.ylabel('\\huge \\% Remaining Privacy Budget')
    plt.xlabel('\\huge $\\epsilon$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    plt.savefig('{}/{}-{}.pdf'.format(output_prefix, dataset_name, 'left-epsilon'))
    plt.clf()


def plot_mean_square_error(k_array, dataset_name, data, output_prefix, theoretical,
                           algorithm_name, baseline_name):
    with open('{}/{}.json'.format(output_prefix, dataset_name), 'w') as f:
        json.dump(data, f)
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
            plt.plot(k_array, improvements, label='\\huge {}'.format(algorithm_name), linewidth=3, markersize=10,
                     marker='o')
            plt.ylim(0, 20)
            plt.ylabel('\\huge \\% Improvement in MSE')
        plt.plot(theoretical_x, 100 * theoretical_y, linewidth=5,
                 linestyle='--', label='\\huge Expected Improvement')
        plt.xlabel('\\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        if epsilon == '0.3':
            logger.info('Figures saved to {}'.format(output_prefix))
            plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',
                                                 str(epsilon).replace('.', '-')))
        plt.clf()

    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    plt.plot(epsilons, improves_for_epsilons, label='\\huge {}'.format(algorithm_name), linewidth=3,
             markersize=10, marker='o')
    plt.plot(epsilons, [100 * theoretical(10) for _ in range(len(epsilons))], linewidth=5,
             linestyle='--', label='\\huge Expected Improvement')
    plt.ylabel('\\huge \\% Improvement in MSE')
    plt.ylim(0, 20)
    plt.xlabel('\\huge $\\epsilon$')
    plt.xticks(np.arange(epsilons.min(), epsilons.max() + 0.1, 0.2))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    plt.savefig('{}/{}-{}-epsilons.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',))
    plt.clf()


def main():
    algorithms = ('All', 'AdaptiveSparseVector', 'GapSparseVector', 'GapTopK')

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(algorithms)))
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    results = arg_parser.parse_args()

    # default value for datasets path
    results.datasets = './datasets' if results.datasets is None else results.datasets

    winning_algorithm = algorithms[
        np.fromiter((difflib.SequenceMatcher(None, results.algorithm, algorithm).ratio() for algorithm in algorithms),
                    dtype=np.float).argmax()
    ]

    winning_algorithm = algorithms[1:] if winning_algorithm == 'All' else (winning_algorithm, )

    output_folder = './figures' if results.output is None else results.output
    k_array = np.fromiter(range(2, 25), dtype=np.int)
    for dataset in process_datasets(results.datasets):
        for algorithm in winning_algorithm:
            # create the output folder if not exists
            algorithm_folder = '{}/{}'.format(os.path.abspath(output_folder), algorithm)
            os.makedirs(algorithm_folder, exist_ok=True)
            output_prefix = os.path.abspath(algorithm_folder)
            plt.hist(dataset[1], bins=200, range=(1, 1000))
            plt.savefig('{}/{}.pdf'.format(output_prefix, dataset[0]))
            plt.savefig(filename)
            plt.clf()

            if 'AdaptiveSparseVector' == algorithm:
                #data = evaluate_adaptivesvt((sparse_vector, adaptive_sparse_vector),
                                            #tuple(epsilon / 10.0 for epsilon in range(1, 16)), dataset)
                with open('/home/grads/ykw5163/mll/refinedp/figures/AdaptiveSparseVector/{}.json'.format(dataset[0]), 'r') as f:
                    data = json.load(f)
                    plot_adaptive(k_array, dataset[0], data, output_prefix)
            if 'GapSparseVector' == algorithm:
                #data = evaluate_gap_estimates((gap_svt_estimates_baseline, gap_svt_estimates),
                                              #tuple(epsilon / 10.0 for epsilon in range(1, 16)), dataset)
                with open('/home/grads/ykw5163/mll/refinedp/figures/GapSparseVector/{}.json'.format(dataset[0]), 'r') as f:
                    data = json.load(f)
                    plot_mean_square_error(k_array, dataset[0], data, output_prefix,
                                           lambda x: 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x))),
                                           'Sparse Vector with Measures', 'gap_svt_estimates_baseline')
            if 'GapTopK' == algorithm:
                with open('/home/grads/ykw5163/mll/refinedp/figures/GapTopK/{}.json'.format(dataset[0]), 'r') as f:
                    data = json.load(f)
                #data = evaluate_gap_estimates((gap_topk_estimates_baseline, gap_topk_estimates),
                                              #tuple(epsilon / 10.0 for epsilon in range(1, 16)), dataset)
                    plot_mean_square_error(k_array, dataset[0], data, output_prefix, lambda x: (x - 1) / (5 * x),
                                           'Noisy Top-K with Measures', 'gap_topk_estimates_baseline')


if __name__ == '__main__':
    main()

