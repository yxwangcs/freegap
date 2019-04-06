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

coloredlogs.install(level='DEBUG', fmt='%(asctime)s %(levelname)s - %(name)s %(message)s')

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
    algorithm_names = ('Classical Sparse Vector', 'Adaptive Sparse Vector with Gap')

    pass


def plot_gaptopk(k_array, dataset_name, data, output_prefix):
    theoretical_x = np.arange(k_array.min(), k_array.max())
    theorectical_y = (theoretical_x - 1) / (5 * theoretical_x)
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict['max_baseline_estimates'])
        for algorithm, algorithm_data in metric_dict.items():
            if algorithm == 'max_baseline_estimates':
                continue
            plt.plot(k_array, 100 * (baseline - np.asarray(algorithm_data)) / baseline,
                     label='\\huge {}'.format('Noisy Top-k with Measures'),
                     linewidth=3, markersize=10)
            plt.ylim(0, 30)
            plt.ylabel('\\huge \\% Improvement in MSE')
            plt.plot(theoretical_x, 100 * theorectical_y, linewidth=3,
                     linestyle='--', label='\\huge \\% Expected Improvement')
        plt.xlabel('\\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        logger.info('Figures saved to {}'.format(output_prefix))
        plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error', epsilon))
        plt.clf()


def plot_gapsvt(k_array, dataset_name, data, output_prefix):
    theoretical_x = np.arange(k_array.min(), k_array.max())
    theorectical_y = 1 / (1 + ((np.power(1 + np.power(2 * theoretical_x, 2.0 / 3), 3)) / theoretical_x * theoretical_x))
    for epsilon, epsilon_dict in data.items():
        assert len(epsilon_dict) == 1 and 'mean_square_error' in epsilon_dict
        metric_dict = epsilon_dict['mean_square_error']
        baseline = np.asarray(metric_dict['svt_baseline_estimates'])
        for algorithm, algorithm_data in metric_dict.items():
            if algorithm == 'svt_baseline_estimates':
                continue
            plt.plot(k_array, 100 * (baseline - np.asarray(algorithm_data)) / baseline,
                     label='\\huge {}'.format('Sparse Vector with with Measures'),
                     linewidth=3, markersize=10)
            plt.ylim(0, 30)
            plt.ylabel('\\huge \\% Improvement in MSE')
            plt.plot(theoretical_x, 100 * theorectical_y, linewidth=3,
                     linestyle='--', label='\\huge \\% Expected Improvement')
        plt.xlabel('\\huge $k$')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)
        plt.gcf().set_tight_layout(True)
        logger.info('Figures saved to {}'.format(output_prefix))
        plt.savefig('{}/{}-{}-{}.pdf'.format(output_prefix, dataset_name, 'Mean_Square_Error', epsilon))
        plt.clf()


def main():
    algorithms = ('All', 'AdaptiveSparseVector', 'RefineLaplace', 'GapSparseVector', 'GapTopK')

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
                                metrics=(above_threshold_answers, precision))
                plot_adaptive(k_array, dataset[0], data, output_prefix)
            if 'GapSparseVector' == algorithm:
                data = evaluate((svt_baseline_estimates, gap_svt_estimates), epsilons, dataset,
                                metrics=(mean_square_error,))
                plot_gapsvt(k_array, dataset[0], data, output_prefix)
            if 'GapTopK' == algorithm:
                data = evaluate((max_baseline_estimates, gap_max_estimates), epsilons, dataset,
                                metrics=(mean_square_error,))
                plot_gaptopk(k_array, dataset[0], data, output_prefix)


if __name__ == '__main__':
    main()

