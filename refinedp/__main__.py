import argparse
import os
import subprocess
import difflib
import logging
import json
import numpy as np
import matplotlib
import shutil
from matplotlib import pyplot as plt
import coloredlogs
from refinedp.adaptivesvt import adaptive_sparse_vector, sparse_vector, evaluate as evaluate_adaptivesvt
from refinedp.gapestimates import gap_svt_estimates, gap_svt_estimates_baseline, \
    gap_topk_estimates, gap_topk_estimates_baseline, evaluate as evaluate_gap_estimates
from refinedp.preprocess import process_t40100k, process_bms_pos, process_kosarak


matplotlib.use('PDF')


# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{sfmath}\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def compress_pdf(file):
    if shutil.which('gs'):
        os.rename(file, '{}.temp'.format(file))
        subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                         '-dPDFSETTINGS=/default',
                         '-dNOPAUSE', '-dQUIET', '-dBATCH',
                         '-sOutputFile={}'.format(file),
                         '{}.temp'.format(file)]
                        )
        os.remove('{}.temp'.format(file))
    else:
        logger.warning('Cannot find Ghost Script executable \'gs\', failed to compress produced PDFs.')


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

    epsilon = '0.7'
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
    adaptive_precision = data[epsilon]['above_threshold_answers']['adaptive_sparse_vector'][quantile]
    plt.plot(k_array, baseline_top_branch,
             label=r'\huge {}'.format('Classical Sparse Vector'),
             linewidth=3, markersize=10, marker='o')
    plt.plot(k_array, adaptive_precision,
             label=r'\huge {}'.format('Adaptive SVT w/ Gap (Total)'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, algorithm_top_branch,
             label=r'\huge {}'.format('Adaptive SVT w/ Gap (Top)'),
             linewidth=3, markersize=10, marker='s')
    plt.plot(k_array, algorithm_middle_branch,
             label=r'\huge {}'.format('Adaptive SVT w/ Gap (Middle)'),
             linewidth=3, markersize=10, marker='^')
    width = 0.6
    plt.ylim(0, 50)
    sub_k_array = np.arange(2, 24, 2)
    colormap = plt.get_cmap('tab10')
    plt.bar(sub_k_array - width, baseline_top_branch[sub_k_array - 1], width, align='edge',
            label=r'\huge Sparse Vector', facecolor=colormap.colors[0] + (0.8,), edgecolor='black', hatch='/')
    #plt.bar(sub_k_array - width, algorithm_total[sub_k_array - 1], width, align='edge',
            #label='\\huge Adaptive SVT w/ Gap  (Total)', facecolor=colormap.colors[1] + (0.8,), hatch='O')
    plt.bar(sub_k_array, algorithm_middle_branch[sub_k_array - 1], width, align='edge', facecolor=colormap.colors[1] + (0.8,),
            edgecolor='black',
            label=r'\huge Adaptive SVT w/ Gap (Middle)', hatch='.')
    plt.bar(sub_k_array, algorithm_top_branch[sub_k_array - 1], width, bottom=algorithm_middle_branch[sub_k_array - 1], align='edge', facecolor=colormap.colors[3] + (0.8,),
            edgecolor='black',
            label=r'\huge Adaptive SVT w/ Gap (Top)', hatch='*')
    plt.ylabel(r'\huge {}'.format(r'\# of Above-Threshold Answers'))
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(sub_k_array)
    legend = plt.legend(framealpha=0, loc=2)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'above_threshold_answers',
                                         str(epsilon).replace('.', '-'))
    plt.savefig(filename)
    compress_pdf(filename)
    plt.clf()

    # plot the precision
    adaptive_precision = data[epsilon]['precision']['adaptive_sparse_vector'][quantile]
    sparse_vector_precision = data[epsilon]['precision']['sparse_vector'][quantile]
    plt.plot(k_array, sparse_vector_precision,
             label=r'\huge {}'.format('Sparse Vector'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.plot(k_array, adaptive_precision,
             label=r'\huge {}'.format('Precision - Adaptive SVT w/ Gap'),
             linewidth=3, markersize=10, marker='P', zorder=5)
    plt.ylim(0, 1.0)
    plt.ylabel(r'\huge {}'.format('Precision'))
    plt.xlabel(r'\huge $k$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'precision',
                                         str(epsilon).replace('.', '-'))
    plt.savefig(filename)
    compress_pdf(filename)
    plt.clf()

    # plot remaining epsilons
    epsilons = np.asarray(tuple(data.keys()), dtype=np.float)
    left_budget = np.asarray(left_epsilons) * 100
    plt.plot(epsilons, left_budget,
             label=r'\huge {}'.format('Adaptive Sparse Vector with Gap'),
             linewidth=3, markersize=10, marker='o')
    plt.ylim(0, 25)
    plt.ylabel(r'\huge \% Remaining Privacy Budget')
    plt.xlabel(r'\huge $\epsilon$')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    legend = plt.legend(loc=3)
    legend.get_frame().set_linewidth(0.0)
    plt.gcf().set_tight_layout(True)
    logger.info('Figures saved to {}'.format(output_prefix))
    filename = '{}/{}-{}.pdf'.format(output_prefix, dataset_name, 'left-epsilon')
    plt.savefig(filename)
    compress_pdf(filename)
    plt.clf()


def plot_mean_square_error(k_array, dataset_name, data, output_prefix, theoretical,
                           algorithm_name, baseline_name):
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
            compress_pdf(filename)
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
    compress_pdf(filename)
    plt.clf()


def main():
    algorithms = {
        'All': (),
        'AdaptiveSparseVector': ((sparse_vector, adaptive_sparse_vector),
                                 evaluate_adaptivesvt, plot_adaptive, {}),
        'GapSparseVector': ((gap_svt_estimates_baseline, gap_svt_estimates),
                            evaluate_gap_estimates, plot_mean_square_error,
                            {'theoretical': lambda x: 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x))),
                             'algorithm_name': 'Sparse Vector with Measures',
                             'baseline_name': 'gap_svt_estimates_baseline'}),
        'GapTopK': ((gap_topk_estimates_baseline, gap_topk_estimates),
                    evaluate_gap_estimates, plot_mean_square_error,
                    {'theoretical': lambda x: (x - 1) / (2 * x),
                     'algorithm_name': 'Noisy Top-K with Measures',
                     'baseline_name': 'gap_topk_estimates_baseline'})
    }

    algorithm_names = tuple(algorithms.keys())

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(algorithms)))
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    arg_parser.add_argument('--clear', help='Clear the output folder', required=False, default=False,
                            action='store_true')
    results = arg_parser.parse_args()

    # default value for datasets path
    results.datasets = './datasets' if results.datasets is None else results.datasets

    winning_algorithm = algorithm_names[
        np.fromiter((difflib.SequenceMatcher(None, results.algorithm, name).ratio() for name in algorithm_names),
                    dtype=np.float).argmax()
    ]

    winning_algorithm = algorithm_names[1:] if winning_algorithm == 'All' else (winning_algorithm, )
    output_folder = os.path.abspath('./figures' if results.output is None else results.output)

    if results.clear:
        logger.info('Clear flag set, removing the output folder...')
        shutil.rmtree(output_folder)

    for dataset in process_datasets(results.datasets):
        for algorithm_name in winning_algorithm:
            # create the algorithm output folder if not exists
            algorithm_folder = os.path.join(output_folder, algorithm_name)
            os.makedirs(algorithm_folder, exist_ok=True)

            # plot statistics figure for dataset
            plt.hist(dataset[1], bins=200, range=(1, 1000))
            filename = os.path.join(algorithm_folder, '{}.pdf'.format(dataset[0]))
            plt.savefig(filename)
            compress_pdf(filename)
            plt.clf()

            # evaluate the algorithms and plot the figures
            evaluate_algorithms, evaluate, plot, kwargs = algorithms[algorithm_name]
            k_array = np.fromiter(range(2, 25), dtype=np.int)

            # check if result json is present (so we don't have to run again)
            # if --clear flag is specified, output folder will be empty, thus won't cause problem here
            json_file = os.path.join(algorithm_folder, '{}-{}.json'.format(algorithm_name, dataset[0]))
            if os.path.exists(json_file):
                logger.info('Found stored json file, loading...')
                with open(json_file, 'r') as fp:
                    data = json.load(fp)
            else:
                logger.info('No json file exists, running experiments...')
                data = evaluate(evaluate_algorithms, tuple(epsilon / 10.0 for epsilon in range(1, 16)), dataset)
                logger.info('Dumping data into json file...')
                with open(json_file, 'w') as fp:
                    json.dump(data, fp)
            logger.info('Plotting')
            plot(k_array, dataset[0], data, algorithm_folder, **kwargs)


if __name__ == '__main__':
    main()

