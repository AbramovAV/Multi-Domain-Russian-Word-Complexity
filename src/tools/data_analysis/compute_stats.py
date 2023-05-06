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
from statsmodels.stats.weightstats import ttest_ind

from src.tools.data_analysis.analysis_utils import (
    aggregate_by_lemma, filter_by_freq_range, load_and_prep_dataframe,
    project_labels_into_discrete, select_labels_for_common_lemmas)
from src.tools.data_preparation import FREQUENCY_RANGES

pd.options.mode.chained_assignment = None


def compute_mean_complexity(dataframe: pd.DataFrame, freq_range=None) -> float:
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


def compute_std_complexity(dataframe: pd.DataFrame, freq_range=None) -> float:
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


def compute_annotator_agreement(
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


def compute_datasets_intersection(
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


def compute_correlation_between_intersection(
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
    label_pairs = select_labels_for_common_lemmas(l_dataframe, r_dataframe)
    p_corr = pearsonr(label_pairs[0],
                      label_pairs[1])
    s_corr = spearmanr(label_pairs[0],
                       label_pairs[1])
    return p_corr, s_corr


def perform_t_test(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        freq_range=None
        ) -> Tuple[float, float, float]:
    """
    Performs Welch t-test with two-sided alternative to test wheter mean
    complexities for two datasets are statistically different from each other
    or not. Optionally, does that only for given frequency range.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.
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
        alternative="two-sided",
        usevar="unequal",
        value=0
    )


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--auxiliary_pools_folder", default=None)
@click.option("--auxiliary_initial_df", default=None)
@click.option("--split_by_freq_ranges", is_flag=True)
# @click.option("--fast_responses_limit", default=15)
def main(
        pools_folder,
        auxiliary_pools_folder,
        auxiliary_initial_df,
        split_by_freq_ranges,
        initial_df,
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
            mean = compute_mean_complexity(dataframe, freq_range)
            std = compute_std_complexity(dataframe, freq_range)
            kappa = compute_annotator_agreement(dataframe, freq_range)
            print("Mean and std complexity:")
            print(np.round(mean, 3), np.round(std, 3))
            print("Fleiss kappa:")
            print(np.round(kappa, 3))
            if auxiliary_pools_folder is not None:
                lemma_intersection = compute_datasets_intersection(
                    dataframe, auxiliary_dataframe, freq_range)
                print("Common lemmas between datasets:")
                print(lemma_intersection)
                _, pval, _ = perform_t_test(
                    dataframe, auxiliary_dataframe, freq_range)
                print("Welch ttest between datasets:")
                print(f"P-value: {pval}")
            print("-" * 20)
    else:
        mean = compute_mean_complexity(dataframe)
        std = compute_std_complexity(dataframe)
        kappa = compute_annotator_agreement(dataframe)
        print(f"Mean complexity: {np.round(mean, 3)}")
        print(f"STD complexity: {np.round(std, 3)}")
        print(f"Fleiss kappa: {np.round(kappa, 3)}")
        if auxiliary_pools_folder is not None:
            lemma_intersection = compute_datasets_intersection(
                dataframe, auxiliary_dataframe)
            print(f"Common lemmas between datasets: {lemma_intersection}")
    if auxiliary_pools_folder is not None:
        p_corr, s_corr = compute_correlation_between_intersection(
            dataframe, auxiliary_dataframe)
        print(f"Pearson corr: {p_corr}, Spearman corr: {s_corr}")
        _, pval, _ = perform_t_test(dataframe, auxiliary_dataframe)
        print("Welch ttest between datasets:")
        print(f"P-value: {pval}")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
