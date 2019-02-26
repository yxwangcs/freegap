import argparse
import sys
import coloredlogs
import logging
import matplotlib
import difflib
import numpy as np
from refinedp.adaptivesvt import evaluate_adaptive_sparse_vector
from refinedp.refinelaplace import evaluate_refine_laplace
from refinedp.gapsvt import evaluate_gap_sparse_vector

# change the matplotlib settings
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},' \
                                             '\\usepackage[libertine]{newtxmath},' \
                                             '\\usepackage{sfmath},' \
                                             '\\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

logger = logging.getLogger(__name__)


def main(argv=sys.argv[1:]):
    options = ('All', 'Adaptive Sparse Vector', 'Refine Laplace', 'Gap Sparse Vector', 'Gap Noisy Max')

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, options are `{}`.'.format(', '.join(options)))
    arg_parser.add_argument('--dataset', help='The dataset folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    results = arg_parser.parse_args(argv)

    kwargs = {'dataset_folder': results.dataset, 'output_folder': results.output}
    # remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    winning_option = options[
        np.fromiter(
            (difflib.SequenceMatcher(None, results.algorithm, option).ratio() for option in options), dtype=np.float)
            .argmax()
    ]

    if winning_option == 'All':
        evaluate_adaptive_sparse_vector(**kwargs)
        evaluate_refine_laplace(**kwargs)
        evaluate_gap_sparse_vector(**kwargs)
    elif winning_option == 'Adaptive Sparse Vector':
        evaluate_adaptive_sparse_vector(**kwargs)
    elif winning_option == 'Refine Laplace':
        evaluate_refine_laplace(**kwargs)
    elif winning_option == 'Gap Sparse Vector':
        evaluate_gap_sparse_vector(**kwargs)


if __name__ == '__main__':
    main()

