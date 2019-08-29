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
from refinedp.adaptivesvt import adaptive_sparse_vector, sparse_vector, \
    evaluate as evaluate_adaptivesvt, plot as plot_adaptive
from refinedp.gapestimates import gap_svt_estimates, gap_svt_estimates_baseline, \
    gap_topk_estimates, gap_topk_estimates_baseline, \
    evaluate as evaluate_gap_estimates, plot as plot_estimates
from refinedp.preprocess import process_t40100k, process_bms_pos, process_kosarak


matplotlib.use('PDF')


# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{sfmath}\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def compress_pdfs(files):
    if shutil.which('gs'):
        for file in files:
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


def main():
    algorithms = {
        'All': (),
        'AdaptiveSparseVector': ((sparse_vector, adaptive_sparse_vector),
                                 evaluate_adaptivesvt, plot_adaptive, {}),
        'GapSparseVector': ((gap_svt_estimates_baseline, gap_svt_estimates),
                            evaluate_gap_estimates, plot_estimates,
                            {'theoretical': lambda x: 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x))),
                             'algorithm_name': 'Sparse Vector with Measures',
                             'baseline_name': 'gap_svt_estimates_baseline'}),
        'GapTopK': ((gap_topk_estimates_baseline, gap_topk_estimates),
                    evaluate_gap_estimates, plot_estimates,
                    {'theoretical': lambda x: (x - 1) / (2 * x),
                     'algorithm_name': 'Noisy Top-K with Measures',
                     'baseline_name': 'gap_topk_estimates_baseline'})
    }

    algorithm_names = tuple(algorithms.keys())

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(algorithms)))
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False, default='./output')
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
    output_folder = os.path.abspath(results.output)

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
            dataset_figure = os.path.join(algorithm_folder, '{}.pdf'.format(dataset[0]))
            plt.savefig(dataset_figure)
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
                data = evaluate(evaluate_algorithms, dataset,
                                epsilons=tuple(epsilon / 10.0 for epsilon in range(1, 16)))
                logger.info('Dumping data into json file...')
                with open(json_file, 'w') as fp:
                    json.dump(data, fp)
            logger.info('Plotting')
            generated_files = plot(k_array, dataset[0], data, algorithm_folder, **kwargs)
            generated_files.append(dataset_figure)
            compress_pdfs(generated_files)


if __name__ == '__main__':
    main()

