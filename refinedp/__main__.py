import argparse
import coloredlogs
import matplotlib
import difflib
from refinedp import *
from refinedp.evaluation import evaluate, mean_square_error, above_threshold_answers
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


def main():
    algorithms = ('All', 'AdaptiveSparseVector', 'RefineLaplace', 'GapSparseVector', 'GapNoisyMax')

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(algorithms)))
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    results = arg_parser.parse_args()

    kwargs = {'output_folder': results.output}
    # remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # default value for datasets path
    results.datasets = './datasets' if results.datasets is None else results.datasets

    winning_algorithm = algorithms[
        np.fromiter((difflib.SequenceMatcher(None, results.algorithm, algorithm).ratio() for algorithm in algorithms),
                    dtype=np.float).argmax()
    ]

    epsilons = (0.3, 0.7, 1.5)

    winning_algorithm = algorithms[1:] if winning_algorithm == 'All' else (winning_algorithm, )

    for dataset in process_datasets(results.datasets):
        if 'AdaptiveSparseVector' in winning_algorithm:
            evaluate((sparse_vector, adaptive_sparse_vector), epsilons, dataset,
                     metrics=(above_threshold_answers, precision),
                     algorithm_names=('Classical Sparse Vector', 'Adaptive Sparse Vector with Gap'), **kwargs)
        if 'GapSparseVector' in winning_algorithm:
            evaluate((svt_baseline_estimates, gap_svt_estimates), epsilons, dataset, metrics=(mean_square_error,),
                     algorithm_names=('Baseline', 'Sparse Vector with Measures'), **kwargs)
        if 'GapNoisyMax' in winning_algorithm:
            evaluate((max_baseline_estimates, gap_max_estimates), epsilons, dataset, metrics=(mean_square_error,),
                     algorithm_names=('Baseline', 'Noisy Top-k with Measures'), **kwargs)


if __name__ == '__main__':
    main()

