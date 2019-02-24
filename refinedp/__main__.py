import numpy as np
import matplotlib.pyplot as plt
import coloredlogs
from refinedp.refinelaplace import refinelaplace
from refinedp.algorithms import adaptive_sparse_vector, sparse_vector
from refinedp.preprocess import *
import matplotlib
from matplotlib import rc
rc('text', usetex=True)

matplotlib.rcParams['text.latex.preamble'] = '\\usepackage[bold]{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc}'

coloredlogs.install(level='INFO', fmt='%(levelname)s - %(name)s %(message)s')


def test_refine_laplace():
    loc, scale, refined_scale = 0, 1, 1 / 2.0
    s = np.random.laplace(loc, scale=scale, size=10000)
    count, bins, ignored = plt.hist(s, 200, density=True, range=(-4., 4))
    x = np.arange(-4., 4., .01)
    pdf = np.exp(-abs(x - loc) / scale) / (2. * scale)
    plt.plot(x, pdf, label='\\huge Laplace($\\mu$=0, scale=1)', linewidth=4)
    plt.title('\\huge $\\mathtt{X: Laplace}$')
    axes = plt.gca()
    axes.set_ylim([0., 1.])
    plt.legend()
    plt.savefig('lap.pdf')
    plt.clf()

    # plot refined laplace
    s = np.fromiter((refinelaplace(elem, 0, 2, 1) for elem in s), dtype=np.float)
    count, bins, ignored = plt.hist(s, 200, density=True, range=(-4., 4))
    refined_pdf = np.exp(-abs(x - loc) / refined_scale) / (2. * refined_scale)
    plt.plot(x, refined_pdf, label='\\huge Laplace($\\mu$=0, scale={})'.format(refined_scale), linewidth=4)
    plt.title('\\huge $\\mathtt{RefineLap}$ (X, 1, 2)', fontsize=30)
    plt.legend()
    plt.savefig('refinelap.pdf')


def plot_dataset(data):
    plt.hist(data[1], 30, density=True, histtype='bar', ec='black', range=(0, 30))

    plt.show()


def compare_SVTs(data, c, epsilon):
    sorted_data = np.sort(data)[::-1]
    threshold = (sorted_data[c] + sorted_data[c + 1]) / 2.0
    print('Threshold: {}'.format(threshold))

    r1, refine_vector = np.asarray(adaptive_sparse_vector(data, threshold, c, epsilon), dtype=np.bool)
    r2 = np.asarray(sparse_vector(data, threshold, c, epsilon), dtype=np.bool)
    print('Total queries: {}'.format(len(data)))
    truth = data > threshold
    print(r1[:len(r2)])
    print(refine_vector[:len(r2)])
    print(r2)

    false_positive_rate_1 = np.count_nonzero(r1 == (truth[:len(r1)] == False)) / float(np.count_nonzero(truth[:len(r1)] == False))
    print('Adaptive sparse vector: {}, trues: {}, false\'s:{}, false positive rate: {}'
          .format(len(r1), np.count_nonzero(r1), np.size(r1) - np.count_nonzero(r1), false_positive_rate_1))

    false_positive_rate_2 = np.count_nonzero(r1 == (truth[:len(r2)] == False)) / float(np.count_nonzero(truth[:len(r2)] == False))
    print('Vanilla sparse vector: {}, trues: {}, false\'s:{}, false positive rate: {}'
          .format(len(r2), np.count_nonzero(r2), np.size(r2) - np.count_nonzero(r2), false_positive_rate_2))

    return c, (false_positive_rate_1, ), (false_positive_rate_2, ), threshold, epsilon


def main():
    test_refine_laplace()


if __name__ == '__main__':
    main()

