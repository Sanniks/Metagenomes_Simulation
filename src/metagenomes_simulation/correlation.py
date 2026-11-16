import scipy.stats as ss

def measure_correlation_coefficient(x, y, correlation_method):
    """Calculate the correlation coefficient using the selected method.
    -Pearson correlation: r = cov(x,y)/(std(x)std(y))
    -Spearman's correlation coefficient: rho = pearson_corralation(rank(x),rank(y))
    where rank(x) transforms the values in the x array into their corresponding ranks in
    the sorted order.
    -Kendall's correlation coefficient: tau = (C - D) / sqrt((C + D + T_x)(C + D + T_y))
    where: C = number of concordant pairs (pairs where x and y change in the
    same direction) D = number of discordant pairs (pairs where x and y
    change in opposite directions) T_x = number of ties only in x T_y =
    number of ties only in y
    
    Args:
        x (array): First set of data.
        y (array): Second set of data.
        correlation_method (str): Correlation method: "Pearson", "Spearman",
            or "Kendall".

    Returns:
        float: Correlation coefficient between x and y. Between -1 and 1.

    Raises:
        `ValueError`: If the correlation method is not valid.
    """
    correlation_method = correlation_method.casefold()
    if correlation_method == "pearson":
        r, _ = ss.pearsonr(x, y)
        return r
    elif correlation_method == "spearman":
        rho, _ = ss.spearmanr(x, y)
        return rho
    elif correlation_method == "kendall":
        tau, _ = ss.kendalltau(x, y)
        return tau
    else:
        raise ValueError(
            "Incorrect value in correlation_method."
            " Only Pearson, Spearman or Kendall are allowed"
            " in correlation_method. "
        )
