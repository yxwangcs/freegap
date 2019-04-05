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
    algorithm_names = ('Baseline', 'Noisy Top-k with Measures')
    # plot and save
    formats = ['-o', '-s']
    markers = ['o', 's', '^']
    for epsilon, epsilon_dict in data.items():
        for metric, metric_dict in epsilon_dict.items():
            for algorithm, algorithm_data in metric_dict.items():
                plt.plot(k_array,
                         100 * (np.asarray(metric_data[epsilon][0][0] - np.asarray(
                             metric_data[epsilon][0][algorithm_index]))) / np.asarray(metric_data[epsilon][0][0]),
                         label='\\huge {}'.format(algorithm_names[algorithm_index]),
                         marker=markers[epsilon_index % len(markers)], linewidth=3,
                         markersize=10)
                plt.ylim(0, 30)
                plt.ylabel('\\huge \\% Improvement in MSE')
            plt.xlabel('\\huge $k$')
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            legend = plt.legend()
            legend.get_frame().set_linewidth(0.0)
            plt.gcf().set_tight_layout(True)
            logger.info('Figures saved to {}'.format(output_prefix))
            plt.savefig('{}/{}-{}-.pdf'.format(output_prefix, dataset_name, metric.replace(' ', '_')))
            plt.clf()


def plot_gapsvt(k_array, dataset_name, data, output_prefix):
    algorithm_names = ('Baseline', 'Sparse Vector with Measures')
    pass


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
            output_folder = '{}/{}'.format(os.path.abspath(output_folder), algorithm)
            os.makedirs(output_folder, exist_ok=True)
            output_prefix = os.path.abspath(output_folder)

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

