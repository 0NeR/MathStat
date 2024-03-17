import numpy as np
import matplotlib.pyplot as plt


def process_list(a_list):
    """
    Process a string of space-separated numbers into a list of floats.
    """
    return list(map(float, a_list.split(' ')))


def compute_cdf(x_list, lambParam):
    """
    Compute cumulative distribution function (CDF) values based on input list and parameter.
    """
    x_list = np.array(x_list)
    cdf = 1 - np.exp(-lambParam * x_list)
    return cdf


def plot_cdf(x_list, cdf):
    """
    Plot cumulative distribution function (CDF).
    """
    for i in range(len(x_list) - 1):
        plt.hlines(cdf[i], x_list[i], x_list[i + 1], colors='b')
        plt.scatter(x_list[i], cdf[i], facecolors='none', edgecolors='b')

    plt.hlines(1, x_list[-1], x_list[-1] * 1.1, colors='b')
    plt.scatter(x_list[-1], 1, facecolors='none', edgecolors='b')
    plt.scatter(x_list[0], cdf[0], color='b')

    plt.xlabel('x', loc='right')
    plt.xticks(np.arange(0, x_list[-1] * 1.1, 0.2))
    plt.xlim(0, x_list[-1] * 1.1)
    plt.ylabel('~_F(x)', loc='top')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylim(0, 1.05)
    plt.grid()
    plt.show()


# Main code
lambParam = 1
a_list = "0.013 0.052 0.076 0.298 0.534 0.635 0.679 0.718 0.791 0.936 0.968 1.039 1.044 1.223 1.279 1.339 1.466 1.600 1.729 1.729 2.104 2.295 3.703 3.749 4.767"

a_list_processed = process_list(a_list)
lambParam = 0.5  # Example parameter value
cdf_values = compute_cdf(a_list_processed, lambParam)
plot_cdf(a_list_processed, cdf_values)
