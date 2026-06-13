import numpy as np
import scipy.stats as ss


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient for two set of data in form of an array.

    r = cov(x,y)/(std(x)std(y))

    Args:
        x (array): First set of data
        y (array): Second set of data

    Returns:
        float : Pearson's correlation coefficient, between -1 and 1
    """
    r, _ = ss.pearsonr(x, y)
    return r


def spearman_correlation(x, y):
    """Calculate spearman's correlation coefficient for two set of data in form of an array.

    rho = pearson_corralation(rank(x),rank(y))

    rank(x) transforms the values in `x` into their corresponding ranks in the sorted order.

    Args:
        x (array): First set of data
        y (array): Second set of data

    Returns:
        float: Spearman's correlation coefficient, between -1 and 1
    """
    rho, _ = ss.spearmanr(x, y)
    return rho


def kendall_correlation(x, y):
    """Calculate kendall's correlation coefficient for two set of data in form of an array.

    tau = (C - D) / sqrt((C + D + T_x)(C + D + T_y))

    where:
        C = number of concordant pairs (pairs where x and y change in the same direction)
        D = number of discordant pairs (pairs where x and y change in opposite directions)
        T_x = number of ties only in x
        T_y = number of ties only in y

    Args:
        x (array): First set of data
        y (array): Second set of data

    Returns:
        float: Spearman's correlation coefficient, between -1 and 1
    """
    tau, _ = ss.kendalltau(x, y)
    return tau
