import argparse
import sys
import coloredlogs
import logging
import matplotlib
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
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('algorithm', help='The algorithm to evaluate, namely '
                                              '`adaptive sparse vector`, `gap sparse vector`, `gap noisy max`, '
                                              '`refine laplace`, or use `all` to evaluate all algorithms.')
    arg_parser.add_argument('--dataset', help='The dataset folder', required=False)
    arg_parser.add_argument('--output', help='The output folder', required=False)
    results = arg_parser.parse_args(argv)

    kwargs = {'dataset_folder': results.dataset, 'output_folder': results.output}
    # remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if 'all' in results.algorithm:
        evaluate_adaptive_sparse_vector(**kwargs)
        evaluate_refine_laplace(**kwargs)
    elif 'adaptive' in results.algorithm:
        evaluate_adaptive_sparse_vector(**kwargs)
    elif 'refine' in results.algorithm:
        evaluate_refine_laplace(**kwargs)
    elif 'gap' in results.algorithm and 'sparse' in results.algorithm:
        evaluate_gap_sparse_vector(**kwargs)
    else:
        print('Invalid algorithm to evaluate.')
        exit(1)


if __name__ == '__main__':
    main()

