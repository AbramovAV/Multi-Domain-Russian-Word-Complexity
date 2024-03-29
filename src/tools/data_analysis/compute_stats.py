"""
Computes multiple statistics for given datasets, such as mean word complexity
and its standard deviation, Fleiss kappa for annotator agreement, Pearson and
Spearman correlation. If given auxiliary dataset for comparison, finds a
number of common lemmas in both and performs Welch t-test with two-sided
alternative.
Optionally, computes all statistic above, except for correlation scores, for
each frequency range among those used for data sampling.
"""

from typing import Tuple

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.weightstats import ttest_ind, ttost_paired

from src.tools.data_analysis.analysis_utils import (
    aggregate_by_lemma, filter_by_freq_range, load_and_prep_dataframe,
    project_labels_into_discrete, select_common_lemmas)
from src.tools.data_preparation import FREQUENCY_RANGES

pd.options.mode.chained_assignment = None


def mean_complexity(dataframe: pd.DataFrame, freq_range=None) -> float:
    """
    Computes mean value for word complexities projected into [0,1] range and
    aggregated by lemma. Optionally, does that only for given freq range.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for filtration.

    Returns:
        mean value in range [0,1]
    """
    dataframe = aggregate_by_lemma(dataframe, freq_range)
    if freq_range is not None:
        dataframe = filter_by_freq_range(dataframe, freq_range)
    return dataframe["OUTPUT:complexity"].mean()


def std_complexity(dataframe: pd.DataFrame, freq_range=None) -> float:
    """
    Computes standard deviation for word complexities projected into [0,1]
    range and aggregated by lemma. Optionally, does that only for given freq
    range.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for filtration.

    Returns:
        standard deviation in range [0,1]
    """
    dataframe = aggregate_by_lemma(dataframe, freq_range)
    if freq_range is not None:
        dataframe = filter_by_freq_range(dataframe, freq_range)
    return dataframe["OUTPUT:complexity"].std()


def annotator_agreement(
        dataframe: pd.DataFrame,
        freq_range=None
        ) -> float:
    """
    https://stats.stackexchange.com/questions/153225/why-does-fleisss-kappa-decrease-with-increased-response-homogeneity/207640#207640

    Computes Fleiss kappa on discrete labels. Optionally, does that only for
    given frequency range.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for filtration.

    Returns:
        Fleiss kappa value in range [-1, 1]. See Wiki for score interpretation
    """
    if freq_range is not None:
        dataframe = filter_by_freq_range(dataframe, freq_range)
    if dataframe["OUTPUT:complexity"].dtype == float:
        dataframe = project_labels_into_discrete(dataframe)
    onehot = pd.get_dummies(dataframe["OUTPUT:complexity"])
    dataframe = pd.concat([dataframe, onehot], axis='columns')
    for k in range(1, 6):
        if k not in dataframe.keys():
            dataframe[k] = 0
    scores_mat = dataframe.groupby("ASSIGNMENT:task_id", sort=False).aggregate(
        {k: "sum" for k in range(1, 6)}
        ).to_numpy(dtype=int)
    return fleiss_kappa(scores_mat, method='fleiss')


def datasets_intersection(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        freq_range=None
        ) -> int:
    """
    Finds how many common lemmas are in both datasets. Optionally, does that
    only for given frequency range.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for filtration.

    Returns:
        number of common lemmas
    """
    l_dataframe = aggregate_by_lemma(l_dataframe, freq_range)
    r_dataframe = aggregate_by_lemma(r_dataframe, freq_range)
    if freq_range:
        l_dataframe = filter_by_freq_range(l_dataframe, freq_range)
        r_dataframe = filter_by_freq_range(r_dataframe, freq_range)
    return len(l_dataframe[l_dataframe.index.isin(r_dataframe.index)])


def correlation_between_intersection(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame
        ) -> Tuple:
    """
    Estimates how well complexities for common lemmas are correlated to each
    other. Works on full dataset only.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.

    Returns:
        Pearson and Spearman correlation scores.
    """
    l_dataframe, r_dataframe = select_common_lemmas(
        l_dataframe,
        r_dataframe
    )
    l_complexity = l_dataframe["OUTPUT:complexity"]
    r_complexity = r_dataframe["OUTPUT:complexity"]
    p_corr = pearsonr(l_complexity,
                      r_complexity)
    s_corr = spearmanr(l_complexity,
                       r_complexity)
    return p_corr, s_corr


def welch_ttest(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        alternative="larger",
        ttest_value=0,
        freq_range=None
        ) -> Tuple[float, float, float]:
    """
    Performs Welch t-test with given alternative to test wheter mean
    complexities for two datasets are statistically different from each other
    or not. Optionally, does that only for given frequency range.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.
        alternative: alternative for ttest - smaller, larger or two-sided.
        ttest_value: difference in means of samples.
        freq_range: frequency range for filtration.

    Returns:
        test statistic value, p-value and number of degrees of freedom.
    """
    l_dataframe = aggregate_by_lemma(l_dataframe)
    r_dataframe = aggregate_by_lemma(r_dataframe)
    if freq_range:
        l_dataframe = filter_by_freq_range(l_dataframe, freq_range)
        r_dataframe = filter_by_freq_range(r_dataframe, freq_range)
    return ttest_ind(
        l_dataframe["OUTPUT:complexity"],
        r_dataframe["OUTPUT:complexity"],
        alternative=alternative,
        usevar="unequal",
        value=ttest_value
    )[1]


def paired_ttost(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        ttost_range=(0, 0.01),
        freq_range=None
        ) -> Tuple[float, float, float]:
    """
    Performs paired t-test with two-sided alternative to test wheter mean
    complexities for two datasets are statistically different from each other
    or not. Optionally, does that only for given frequency range.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.
        ttost_range: lower and upper values for tests.
        freq_range: frequency range for filtration.

    Returns:
        test statistic value, p-value and number of degrees of freedom.
    """
    if freq_range is not None:
        l_dataframe = filter_by_freq_range(l_dataframe, freq_range)
        r_dataframe = filter_by_freq_range(r_dataframe, freq_range)
    l_dataframe, r_dataframe = select_common_lemmas(
        l_dataframe,
        r_dataframe
    )
    l_complexity = l_dataframe["OUTPUT:complexity"]
    r_complexity = r_dataframe["OUTPUT:complexity"]
    ppval, (_, l_pval, _), (_, u_pval, _) = ttost_paired(
                                                         l_complexity,
                                                         r_complexity,
                                                         low=ttost_range[0],
                                                         upp=ttost_range[1],
                                                        )
    return ppval, l_pval, u_pval


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--auxiliary_pools_folder", default=None)
@click.option("--auxiliary_initial_df", default=None)
@click.option("--split_by_freq_ranges", is_flag=True)
@click.option("--ttest_alternative",
              default="larger",
              type=click.Choice(
                  ['larger', 'smaller', 'two_sided'],
                  case_sensitive=True))
@click.option("--ttest_value", default=0, type=float)
@click.option("--ttost_range", type=(float, float), default=(0, 0.01))
# @click.option("--fast_responses_limit", default=15)
def main(  # pylint: disable=too-many-arguments
        pools_folder,
        auxiliary_pools_folder,
        auxiliary_initial_df,
        split_by_freq_ranges,
        initial_df,
        ttest_alternative,
        ttest_value,
        ttost_range
        # fast_responses_limit
        ) -> None:
    """
    POOLS_FOLDER: directory with annotation results (tsv) from toloka
    INITIAL_DF: tsv file with all sentences and their data (lemma, freq, word)
    """
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    if auxiliary_pools_folder is not None:
        auxiliary_dataframe = load_and_prep_dataframe(
            auxiliary_pools_folder,
            auxiliary_initial_df)
    if split_by_freq_ranges:
        for freq_range in sorted(FREQUENCY_RANGES):
            print(f"For freq range {freq_range[0]}-{freq_range[1]} ipm:")
            print("-" * 20)
            print("Mean and std complexity:")
            print(
                np.round(mean_complexity(dataframe, freq_range), 3),
                np.round(std_complexity(dataframe, freq_range), 3))
            print("Fleiss kappa:")
            print(np.round(
                annotator_agreement(dataframe, freq_range), 3))
            if auxiliary_pools_folder is not None:
                print("Common lemmas between datasets:")
                print(datasets_intersection(
                    dataframe, auxiliary_dataframe, freq_range))
                pval = welch_ttest(
                    dataframe, auxiliary_dataframe, ttest_alternative,
                    ttest_value, freq_range)
                print("Welch ttest between datasets:")
                print(f"P-value: {pval}")
                ppvals = paired_ttost(
                    dataframe, auxiliary_dataframe, freq_range, ttost_range)
                print("Paired TOST between datasets:")
                print(f"P-value: {ppvals[0]}, p-value (lower): {ppvals[1]}, "
                      f"p-value (upper): {ppvals[2]}")
            print("-" * 20)
    else:
        print(f"Mean complexity: {np.round(mean_complexity(dataframe), 3)}")
        print(f"STD complexity: {np.round(std_complexity(dataframe), 3)}")
        print(f"Fleiss kappa: {np.round(annotator_agreement(dataframe), 3)}")
        if auxiliary_pools_folder is not None:
            lemma_intersection = datasets_intersection(
                dataframe, auxiliary_dataframe)
            print(f"Common lemmas between datasets: {lemma_intersection}")
    if auxiliary_pools_folder is not None:
        corrs = correlation_between_intersection(
            dataframe, auxiliary_dataframe)
        print(f"Pearson corr: {corrs[0]}, Spearman corr: {corrs[1]}")
        pval = welch_ttest(
                    dataframe, auxiliary_dataframe, ttest_alternative,
                    ttest_value)
        print("Welch ttest between datasets:")
        print(f"P-value: {pval}")
        ppvals = paired_ttost(
            dataframe, auxiliary_dataframe, ttost_range)
        print("Paired TOST between datasets:")
        print(f"P-value: {ppvals[0]}, p-value (lower): {ppvals[1]}, "
              f"p-value (upper): {ppvals[2]}")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
