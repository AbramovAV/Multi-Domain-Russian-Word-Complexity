from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa

from src.tools.data_analysis.analysis_utils import filter_by_freq_range, aggregate_by_lemma, filter_by_fast_responses, \
              project_labels_into_discrete, load_and_prep_dataframe
from src.tools.data_preparation.prepare_data_for_annotation import FREQUENCY_RANGES

pd.options.mode.chained_assignment = None

def compute_mean_complexity(dataframe:pd.DataFrame, freq_range=None) -> float:
    dataframe = aggregate_by_lemma(dataframe, freq_range)
    if freq_range is not None:
        dataframe = filter_by_freq_range(dataframe, freq_range)
    return dataframe["OUTPUT:complexity"].mean()


def compute_std_complexity(dataframe:pd.DataFrame, freq_range=None):
    dataframe = aggregate_by_lemma(dataframe, freq_range)
    if freq_range is not None:
        dataframe = filter_by_freq_range(dataframe, freq_range)
    return dataframe["OUTPUT:complexity"].std()


def compute_annotator_agreement(dataframe, freq_range=None):
    """
    https://stats.stackexchange.com/questions/153225/why-does-fleisss-kappa-decrease-with-increased-response-homogeneity/207640#207640
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
        {k:"sum" for k in range(1,6)}
        ).to_numpy(dtype=int)
    return fleiss_kappa(scores_mat, method='fleiss')    


def compute_datasets_intersection(l_dataframe:pd.DataFrame, r_dataframe:pd.DataFrame, freq_range=None) -> int:
    l_dataframe = aggregate_by_lemma(l_dataframe, freq_range)
    r_dataframe = aggregate_by_lemma(r_dataframe, freq_range)
    if freq_range:
        l_dataframe = filter_by_freq_range(l_dataframe, freq_range)
        r_dataframe = filter_by_freq_range(r_dataframe, freq_range)
    return len(l_dataframe[l_dataframe.index.isin(r_dataframe.index)])


def compute_correlation_between_intersection(l_dataframe:pd.DataFrame, r_dataframe:pd.DataFrame) -> int:
    l_dataframe = aggregate_by_lemma(l_dataframe)
    r_dataframe = aggregate_by_lemma(r_dataframe)
    l_inter_ids = l_dataframe.index.isin(r_dataframe.index)
    r_inter_ids = r_dataframe.index.isin(l_dataframe.index)
    l_complexity = l_dataframe[l_inter_ids].sort_index()["OUTPUT:complexity"]
    r_complexity = r_dataframe[r_inter_ids].sort_index()["OUTPUT:complexity"]
    p_corr = pearsonr(l_complexity,
                      r_complexity)
    s_corr = spearmanr(l_complexity,
                       r_complexity)
    return p_corr, s_corr


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--auxiliary_pools_folder", default=None)
@click.option("--auxiliary_initial_df", default=None)
@click.option("--split_by_freq_ranges", is_flag=True)
@click.option("--fast_responses_limit", default=15)
def main(pools_folder, auxiliary_pools_folder, auxiliary_initial_df, split_by_freq_ranges, initial_df, fast_responses_limit):
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    if auxiliary_pools_folder is not None:
        auxiliary_dataframe = load_and_prep_dataframe(auxiliary_pools_folder, auxiliary_initial_df)
    if split_by_freq_ranges:
        for freq_range in sorted(FREQUENCY_RANGES):
            mean = compute_mean_complexity(dataframe, freq_range)
            std = compute_std_complexity(dataframe, freq_range)
            kappa = compute_annotator_agreement(dataframe, freq_range)
            print(f"Mean and std complexity for freq range {freq_range[0]}-{freq_range[1]} ipm:")
            print(np.round(mean, 3), np.round(std, 3))
            print(f"Fleiss kappa for freq range {freq_range[0]}-{freq_range[1]} ipm:")
            print(np.round(kappa, 3))
            if auxiliary_pools_folder is not None:
                lemma_intersection = compute_datasets_intersection(dataframe, auxiliary_dataframe, freq_range)
                print(f"Intersected lemmas between datasets for freq range {freq_range[0]}-{freq_range[1]} ipm:")
                print(lemma_intersection)
    else:
        mean = compute_mean_complexity(dataframe)
        std = compute_std_complexity(dataframe)
        kappa = compute_annotator_agreement(dataframe)
        print(f"Overall mean complexity: {np.round(mean, 3)}")
        print(f"Overall std complexity: {np.round(std, 3)}")
        print(f"Fleiss kappa: {np.round(kappa, 3)}")
        if auxiliary_pools_folder is not None:
            lemma_intersection = compute_datasets_intersection(dataframe, auxiliary_dataframe)
            print(f"Overall intersected lemmas between datasets: {lemma_intersection}")
    if auxiliary_pools_folder is not None:
        p_corr, s_corr = compute_correlation_between_intersection(dataframe, auxiliary_dataframe)
        print(f"Pearson corr: {p_corr}, Spearman corr: {s_corr}")
if __name__ == '__main__':
    main()

