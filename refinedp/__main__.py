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


def process_datasets(folder):
    logger.info('Loading datasets')
    dataset_folder = os.path.abspath(folder)
    # yield different datasets with their names
    yield 'T40100K', process_t40100k('{}/T40I10D100K.dat'.format(dataset_folder))
    yield 'SF1', process_sf1('{}/DEC_10_SF1_PCT3.csv'.format(dataset_folder))
    yield 'BMS-POS', process_bms_pos('{}/BMS-POS.dat'.format(dataset_folder))
    yield 'kosarak', process_kosarak('{}/kosarak.dat'.format(dataset_folder))


def plot_adaptive(k_array, dataset_name, data, output_prefix):
    with open('{}/{}.json'.format(output_prefix, dataset_name), 'w') as f:
        json.dump(data, f)
    algorithm_names = ('Classical Sparse Vector', 'Adaptive Sparse Vector with Gap')

    pass


def plot_mean_square_error(k_array, dataset_name, data, output_prefix, theoretical,
                           algorithm_name, baseline_name):
    with open('{}/{}.json'.format(output_prefix, dataset_name), 'w') as f:
        json.dump(data, f)
    theoretical_x = np.arange(k_array.min(), k_array.max())
    theoretical_y = theoretical(theoretical_x)
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict[baseline_name])
        for algorithm, algorithm_data in metric_dict.items():
            if algorithm == baseline_name:
                continue
            plt.plot(k_array, 100 * (baseline - np.asarray(algorithm_data)) / baseline,
                     label='\\huge {}'.format(algorithm_name),
                     linewidth=3, markersize=10, marker='o')
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
        logger.info('Figures saved to {}'.format(output_prefix))
        plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error',
                                             str(epsilon).replace('.', '-')))
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

    epsilons = (0.3, 0.7, 1.5)

    winning_algorithm = algorithms[1:] if winning_algorithm == 'All' else (winning_algorithm, )

    output_folder = './figures' if results.output is None else results.output
    k_array = np.fromiter(range(2, 25), dtype=np.int)
    for dataset in process_datasets(results.datasets):
        for algorithm in winning_algorithm:
            # create the output folder if not exists
            algorithm_folder = '{}/{}'.format(os.path.abspath(output_folder), algorithm)
            os.makedirs(algorithm_folder, exist_ok=True)
            output_prefix = os.path.abspath(algorithm_folder)

            if 'AdaptiveSparseVector' == algorithm:
                data = evaluate((sparse_vector, adaptive_sparse_vector), epsilons, dataset,
                                metrics=(top_branch, middle_branch, top_branch_precision, middle_branch_precision))
                plot_adaptive(k_array, dataset[0], data, output_prefix)
            if 'GapSparseVector' == algorithm:
                data = evaluate_gap_estimates((gap_svt_estimates_baseline, gap_svt_estimates), (0.3, 0.7, 1.5), dataset)
                plot_mean_square_error(k_array, dataset[0], data, output_prefix,
                                       lambda x: 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x))),
                                       'Sparse Vector with Measures', 'gap_svt_estimates_baseline')
            if 'GapTopK' == algorithm:
                data = evaluate_gap_estimates((gap_topk_estimates_baseline, gap_topk_estimates), (0.3, 0.7, 1.5), dataset)
                plot_mean_square_error(k_array, dataset[0], data, output_prefix, lambda x: (x - 1) / (5 * x),
                                       'Noisy Top-K with Measures', 'gap_topk_estimates_baseline')


if __name__ == '__main__':
    main()

