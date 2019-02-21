import numpy as np
import matplotlib.pyplot as plt
from refinedp.refinelaplace import refinelaplace


def test_refine_laplace():
    loc, scale, refined_scale = 0, 1, 1 / 2.0
    s = np.random.laplace(loc, scale=scale, size=10000)
    count, bins, ignored = plt.hist(s, 200, density=True, range=(-4., 4))
    x = np.arange(-4., 4., .01)
    pdf = np.exp(-abs(x - loc) / scale) / (2. * scale)
    plt.plot(x, pdf, label='Laplace($\\mu$=0, scale=1)')
    plt.title('$\mathtt{Laplace}$')
    axes = plt.gca()
    axes.set_ylim([0., 1.])
    plt.legend()
    plt.savefig('lap.svg')
    plt.clf()

    # plot refined laplace
    s = np.fromiter((refinelaplace(elem, 0, 2, 1) for elem in s), dtype=np.float)
    count, bins, ignored = plt.hist(s, 200, density=True, range=(-4., 4))
    refined_pdf = np.exp(-abs(x - loc) / refined_scale) / (2. * refined_scale)
    plt.plot(x, refined_pdf, label='Laplace($\\mu$=0, scale=2)')
    plt.title('$\mathtt{RefineLap}$(X, 2, 1)')
    plt.legend()
    plt.savefig('refinelap.svg')


def compare_SVTs():
    q = np.asarray([1 for _ in range(2000)] + [10000 for _ in range(18000)])
    np.random.shuffle(q)
    threshold = 10
    r1 = np.asarray(adaptive_sparse_vector(q, threshold, 40, 0.7), dtype=np.bool)
    r2 = np.asarray(sparse_vector(q, threshold, 40, 0.7), dtype=np.bool)
    print('Total queries: {}'.format(len(q)))
    truth = q > 1.5
    print('Adaptive sparse vector: {}, trues: {}, false\'s:{}, accuray: {}'
          .format(len(r1), np.count_nonzero(r1), np.size(r1) - np.count_nonzero(r1),
                  np.count_nonzero(r1 == truth[:len(r1)]) / len(r1)))
    print('Vanilla sparse vector: {}, trues: {}, false\'s:{}, accuray: {}'
          .format(len(r2), np.count_nonzero(r2), np.size(r2) - np.count_nonzero(r2),
                  np.count_nonzero(r2 == truth[:len(r2)]) / len(r2)))




def main():
    test_refine_laplace()


if __name__ == '__main__':
    main()

