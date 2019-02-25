import numpy as np
import matplotlib.pyplot as plt
import coloredlogs
from refinedp.refinelaplace import refinelaplace
from refinedp.algorithms import adaptive_sparse_vector, sparse_vector
from refinedp.preprocess import *
import matplotlib
from matplotlib import rc
rc('text', usetex=True)

matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(levelname)s - %(name)s %(message)s')


def test_refine_laplace():
    loc, scale, refined_scale = 0, 1, 1 / 2.0
    s = np.random.laplace(loc, scale=scale, size=10000)
    count, bins, ignored = plt.hist(s, 150, density=True, range=(-4., 4))
    x = np.arange(-4., 4., .01)
    pdf = np.exp(-abs(x - loc) / scale) / (2. * scale)
    plt.plot(x, pdf, label='\\huge Laplace($\\mu$=0, scale=1)', linewidth=3)
    plt.title('\\huge X: Laplace')
    axes = plt.gca()
    axes.set_ylim([0., 1.])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.savefig('lap.pdf')
    plt.clf()

    # plot refined laplace
    s = np.fromiter((refinelaplace(elem, 0, 2, 1) for elem in s), dtype=np.float)
    count, bins, ignored = plt.hist(s, 150, density=True, range=(-4., 4))
    refined_pdf = np.exp(-abs(x - loc) / refined_scale) / (2. * refined_scale)
    plt.plot(x, refined_pdf, label='\\huge Laplace($\\mu$=0, scale={})'.format(refined_scale), linewidth=3)
    plt.title('\\huge RefineLap (X, 1, 2)', fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.savefig('refinelap.pdf')


def plot_dataset(data):
    plt.hist(data[1], 30, density=True, histtype='bar', ec='black', range=(0, 30))
    plt.show()


def calc_metrics(data, answer, truth):
    # answer_rate, false_negative_rate
    return float(len(answer)) / len(data),\
           float(np.count_nonzero(answer == (truth[:len(answer)] == False))) / np.count_nonzero(truth[:len(answer)] == False)


def compare_SVTs(data, c, epsilon):
    sorted_data = np.sort(data)[::-1]
    threshold = (sorted_data[c] + sorted_data[c + 1]) / 2.0

    r1 = np.asarray(adaptive_sparse_vector(data, threshold, c, epsilon), dtype=np.bool)
    r2 = np.asarray(sparse_vector(data, threshold, c, epsilon), dtype=np.bool)

    truth = data > threshold

    return c, calc_metrics(data, r1, truth), calc_metrics(data, r2, truth), threshold, epsilon


def plot_metrics(results):
    METRICS = ['Answer Rate', 'False Negative Rate']
    # [ [[adaptive_metric_1], [adaptive_metric_1_err], [vanilla_metric_1, vanilla_metric_1_err]],
    #   [metric_2 ...],
    #   [metric_3 ...],
    #   ...
    # ]
    plot_data = [[[], [], [], []] for _ in range(len(METRICS))]
    for name, data in results.items():
        c_array = list(data.keys())
        for c, stats in data.items():
            for i in range(len(METRICS)):
                # adaptive sparse vector
                metric_data = np.fromiter((x[1][i] for x in stats), dtype=np.float)
                plot_data[i][0].append(metric_data.mean())
                plot_data[i][1].append([metric_data.mean() - metric_data.min(), metric_data.max() - metric_data.mean()])

                # vanilla sparse vector
                metric_data = np.fromiter((x[2][i] for x in stats), dtype=np.float)
                plot_data[i][2].append(metric_data.mean())
                plot_data[i][3].append([metric_data.mean() - metric_data.min(), metric_data.max() - metric_data.mean()])

        for index, metric in enumerate(METRICS):
            plt.errorbar(c_array, plot_data[index][0], yerr=np.transpose(plot_data[index][1]), label='\\huge Adaptive Sparse Vector', fmt='-o', markersize=8)
            plt.errorbar(c_array, plot_data[index][2], yerr=np.transpose(plot_data[index][3]), label='\\huge Sparse Vector', fmt='-s', markersize=8)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend()
            plt.title('{} - {}'.format(name, metric), fontsize=24)
            plt.savefig('{}_{}.pdf'.format(name, metric))
            plt.clf()


def main():
    test_refine_laplace()
    datasets = {
    }

    results = {}
    for name, data in datasets.items():
        print('Evaluating on {}...'.format(name))
        plot_dataset(data)
        results[name] = {}
        for c in range(25, 325, 25):
            results[name][c] = []
            for _ in range(5):
                stats = compare_SVTs(data[1], c, 0.3)
                results[name][c].append(stats)
    plot_metrics(results)


if __name__ == '__main__':
    main()

