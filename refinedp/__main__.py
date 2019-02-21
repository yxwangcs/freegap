import numpy as np
import matplotlib.pyplot as plt
from refinedp.refinelaplace import refinelaplace


def test_refine_laplace():
    loc, scale = 0, 1
    new_scale = 1 / 1.5
    s = np.random.laplace(loc, scale=scale, size=10000)
    s = np.fromiter((refinelaplace(elem, 0, 1.5, 1) for elem in s), dtype=np.float)
    count, bins, ignored = plt.hist(s, 200, density=True)
    x = np.arange(-8., 8., .01)
    pdf = np.exp(-abs(x - loc) / scale) / (2. * scale)
    new_pdf = pdf = np.exp(-abs(x - loc) / new_scale) / (2. * new_scale)
    # plt.plot(x, pdf)
    plt.plot(x, new_pdf)
    #plt.show()
    plt.savefig('noise_down.svg')


def main():
    test_refine_laplace()


if __name__ == '__main__':
    main()

