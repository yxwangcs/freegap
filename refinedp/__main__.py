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
from refinedp.adaptivesvt import adaptive_sparse_vector, \
    top_branch, top_branch_precision, middle_branch, middle_branch_precision, precision, f_measure, \
    above_threshold_answers, plot as plot_adaptive
from refinedp.gapestimates import gap_svt_estimates, gap_topk_estimates, mean_square_error, plot as plot_estimates
from refinedp.preprocess import process_t40100k, process_bms_pos, process_kosarak
from refinedp.evaluate import evaluate

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
        'AdaptiveSparseVector': (adaptive_sparse_vector,
                                 (top_branch, top_branch_precision, middle_branch, middle_branch_precision, precision,
                                  f_measure, above_threshold_answers),
                                 plot_adaptive, {}),
        'GapSparseVector': (gap_svt_estimates, (mean_square_error, ), plot_estimates,
                            {# counting queries
                             'theoretical': lambda x: 1 / (1 + ((np.power(1 + np.power(x, 2.0 / 3), 3)) / (x * x))),
                            #'theoretical': lambda x: 1 / (1 + ((np.power(1 + np.power(2 * x, 2.0 / 3), 3)) / (x * x))),
                             'algorithm_name': 'Sparse Vector with Measures'}),
        'GapTopK': (gap_topk_estimates, (mean_square_error, ), plot_estimates,
                    {# counting queries
                     'theoretical': lambda x: (x - 1) / (2 * x),
                    # 'theoretical': lambda x: (x - 1) / (5 * x)
                     'algorithm_name': 'Noisy Top-K with Measures'})
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

    for algorithm_name in winning_algorithm:
        # create the algorithm output folder if not exists
        algorithm_folder = os.path.join(output_folder, algorithm_name)
        if results.clear:
            logger.info('Clear flag set, removing the algorithm output folder...')
            shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(algorithm_folder, exist_ok=True)

        for dataset in process_datasets(results.datasets):
            # plot statistics figure for dataset
            plt.hist(dataset[1], bins=200, range=(1, 1000))
            dataset_figure = os.path.join(algorithm_folder, '{}.pdf'.format(dataset[0]))
            plt.savefig(dataset_figure)
            plt.clf()

            # evaluate the algorithms and plot the figures
            evaluate_algorithm, metrics, plot, kwargs = algorithms[algorithm_name]
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
                data = evaluate(evaluate_algorithm, dataset, metrics=metrics,
                                epsilons=tuple(epsilon / 10.0 for epsilon in range(1, 16)), k_array=k_array)
                logger.info('Dumping data into json file...')
                with open(json_file, 'w') as fp:
                    json.dump(data, fp)
            logger.info('Plotting')
            generated_files = plot(k_array, dataset[0], data, algorithm_folder, **kwargs)
            generated_files.append(dataset_figure)
            compress_pdfs(generated_files)


if __name__ == '__main__':
    main()

