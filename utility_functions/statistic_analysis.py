from typing import Dict, List
import scipy.stats as stats
import itertools
import numpy as np
import matplotlib.pyplot as plt


def paired_t_test_statistics(samples: Dict[str, List], alpha=0.05):
    # all possible pairings
    n = len(samples)
    pairings = tuple(prod for prod in itertools.product(samples, samples))
    name_pairs = [(pair[0][0], pair[1][0]) for pair in pairings]
    paired_t_test = [stats.ttest_rel(pair[0][1], pair[1][1]) for pair in pairings]
    matrix = [1 if (p_v := pair.pvalue) <= alpha else 0 for pair in paired_t_test]
    return name_pairs, paired_t_test, np.array(matrix).reshape((n, n))


def plot_t_test_matrix(mat, ticks):
    mat_ = mat.copy().astype(np.float32)
    for i in range(mat.shape[0]):
        mat_[i, i] = 0.5
    fig, ax = plt.subplots()
    ax.imshow(mat_, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation='vertical')
    ax.set_yticks(range(len(ticks)))
    ax.set_yticklabels(ticks, rotation='horizontal')
