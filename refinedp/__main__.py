import argparse
import sys
import coloredlogs
import logging
import matplotlib
import difflib
import numpy as np
from refinedp import evaluate, process_datasets


# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},' \
                                             '\\usepackage[libertine]{newtxmath},' \
                                             '\\usepackage{sfmath},' \
                                             '\\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def main(argv=sys.argv[1:]):
    options = ('All', 'AdaptiveSparseVector', 'RefineLaplace', 'GapSparseVector', 'GapNoisyMax')

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(options)))
    arg_parser.add_argument('--datasets', help='The datasets folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    results = arg_parser.parse_args(argv)

    kwargs = {'output_folder': results.output}
    # default value for datasets path
    results.datasets = './datasets' if results.datasets is None else results.datasets
    # remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    winning_option = options[
        np.fromiter((difflib.SequenceMatcher(None, results.algorithm, option).ratio() for option in options),
                    dtype=np.float).argmax()
    ]

    for dataset in process_datasets(results.dataset):
        if winning_option == 'All':
            evaluate_adaptive_sparse_vector(**kwargs)
            evaluate_refine_laplace(**kwargs)
            evaluate_gap_sparse_vector(**kwargs)
        elif winning_option == 'AdaptiveSparseVector':
            evaluate_adaptive_sparse_vector(**kwargs)
        elif winning_option == 'RefineLaplace':
            evaluate_refine_laplace(**kwargs)
        elif winning_option == 'GapSparseVector':
            evaluate_gap_sparse_vector(**kwargs)
        elif winning_option == 'GapNoisyMax':
            evaluate_gap_k_noisy_max()


if __name__ == '__main__':
    main()

