"""
Functions to visualise results analysis
"""
import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

def qqplot(pvalues, log=True, ax=None, marker="o", color=None, alpha=0.8, 
           title=None, xlabel=None, ylabel=None, ablinecolor="r", is_show=None, 
           dpi=300, figname=None, **kwargs):
    """
    Generates QQ-plot using p-values to check for systemic bias in results. 

    Parameters
        ------------
        pvalues: list/array,
            list of p-values
        Returns
        ------------
        datadict: dictionary
            dictionary containing the "data" and "info", which are genotype data and their information respectively.
    """

    if xlabel is None:
        xlabel = r"$Expected(-log_{10}{(P)})$" if other is None else r"$-log_{10}{(Value)} of 2nd Sample$"
    if ylabel is None:
        ylabel = r"$Observed(-log_{10}{(P)})$" if other is None else r"(-log_{10}{(Value)}) of 1st Sample$"

    data = np.array(data, dtype=float)

    # create observed and expected
    e = ppoints(len(data)) if other is None else sorted(other)

    if logp:
        o = -np.log10(sorted(data))
        e = -np.log10(e)
    else:
        o = np.array(sorted(data))
        e = np.array(e)

    if "marker" not in kwargs:
        kwargs["marker"] = marker
    ax = _do_plot(e, o, ax=ax, color=color, ablinecolor=ablinecolor, alpha=alpha, **kwargs)

    ## Compute the genomic inflation factor, lambda
    ## The genomic inflation factor was defined as the median of the observed chi-squared test statistics divided by the expected median of the corresponding chi-squared distribution, from https://onlinelibrary.wiley.com/doi/full/10.1111/jbg.12419
    ## ppf used to obtain the test statistic, more can be learned here https://stackoverflow.com/questions/65468026/norm-ppf-vs-norm-cdf-in-pythons-scipy-stats
    expected_median = chi2.ppf(0.5, 1)  # 0.4549364 ~ 0.456
    lambda_value = round(np.median(chi2.ppf(1-pvalues, 1)) / expected_median, 3)


    if title:
        title += r"$(\lambda = %s)$" % lambda_value
    else:
        title = r"$\lambda = %s$" % lambda_value

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if (is_show is None) and (figname is None):
        is_show = True

    General.get_figure(is_show, fig_name=figname, dpi=dpi)
    return ax