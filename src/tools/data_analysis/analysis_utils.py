"""
A collection of utility functions used during data analysis and visualization.
Loads data, aggregates it by lemma or annotation task id, transforms word
complexity labels into continuous and discrete range, and filters data.
"""
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.tools.data_preparation.prepare_data_for_annotation import _add_marks


def merge_annotated_toloka_tsv(
        *pathes: List[str | Path],
        drop_cols=None
        ) -> pd.DataFrame:
    """
    Joins several tsv files with Toloka annotations into single dataframe.
    Optionally, removes unnecessary columns.

    Args:
        pathes: list of pathes to tsv files.
        drop_cols: list of column titles to remove.

    Returns:
        pandas Dataframe with word complexity, contexts, annotation metadata.
    """
    dataframe = pd.read_csv(pathes[0], sep='\t')
    for path in pathes[1:]:
        next_dataframe = pd.read_csv(path, sep='\t')
        dataframe = pd.concat([dataframe, next_dataframe], ignore_index=True)
    if drop_cols is not None:
        dataframe = dataframe.drop(drop_cols, axis="columns")
    return dataframe


def add_freq_for_sentence(
        dataframe: pd.DataFrame,
        initial_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds information about word frequency by loading and
    left-joining dataframe with original data sampled before annotation.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        initial_dataframe: original data with contexts, lemmas, and words.

    Returns:
        same Dataframe with added columns frequency(ipm) and lemma.
    """
    prep_contexts = initial_dataframe.apply(
            _add_marks,
            axis=1
    )
    initial_dataframe['context'] = prep_contexts
    dataframe = pd.merge(dataframe, initial_dataframe, how='left',
                         left_on='INPUT:text', right_on='context')
    dataframe = dataframe.drop(labels=["start_idx", "context"], axis="columns")
    return dataframe


def project_labels_into_contunious(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms discrete complexity labels from Likert scale into continuous.

    Args:
        dataframe: pandas Dataframe with discrete complexity labels.

    Returns:
        same Dataframe, but with continuous labels in range [0, 1].
    """
    dataframe = dataframe.copy(deep=True)
    dataframe["OUTPUT:complexity"] -= 1
    dataframe["OUTPUT:complexity"] *= 0.25
    return dataframe


def project_labels_into_discrete(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms continuous complexity labels into discrete from Likert scale.

    Args:
        dataframe: pandas Dataframe with continuous complexity labels.

    Returns:
        same Dataframe, but with discrete complexity labels 1-5.
    """
    dataframe = dataframe.copy(deep=True)
    dataframe["OUTPUT:complexity"] = (
        (dataframe["OUTPUT:complexity"] / 0.25) + 1
        ).astype(int)
    return dataframe


def aggregate_by_task(
        dataframe: pd.DataFrame,
        freq_range=None,
        auxiliary_mapping=None
        ) -> pd.DataFrame:
    """
    Aggregates Dataframe with word complexity info by Toloka task ids.
    Each aggregated record has mean word complexity and corresponding lemma.
    Optionally, aggregates keys from auxiliary_mapping.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for aggregation.
        auxiliary_mapping: additional columns and actions for aggregation.

    Returns:
        aggregated Dataframe with ASSIGNMENT:task_id as index.
    """
    if auxiliary_mapping is None:
        auxiliary_mapping = {}
    dataframe = dataframe.groupby("ASSIGNMENT:task_id", sort=False)
    keys = {"OUTPUT:complexity": "mean", "lemma": "first"} | auxiliary_mapping
    if freq_range is not None:
        keys["frequency(ipm)"] = "mean"
    dataframe = dataframe.aggregate(keys)
    return dataframe


def aggregate_by_lemma(
        dataframe: pd.DataFrame,
        freq_range=None,
        auxiliary_mapping=None
        ) -> pd.DataFrame:
    """
    Aggregates Dataframe with word complexity info by lemma.
    Each aggregated record has mean word complexity and corresponding lemma.
    Optionally, aggregates keys from auxiliary_mapping.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for aggregation.
        auxiliary_mapping: additional columns and actions for aggregation.

    Returns:
        aggregated Dataframe with lemma as index.
    """
    if auxiliary_mapping is None:
        auxiliary_mapping = {}
    dataframe = dataframe.groupby("lemma", sort=False)
    keys = {"OUTPUT:complexity": "mean", "lemma": "first"} | auxiliary_mapping
    if freq_range is not None:
        keys["frequency(ipm)"] = "mean"
    dataframe = dataframe.aggregate(keys)
    return dataframe


def filter_by_freq_range(
        dataframe: pd.DataFrame,
        freq_range=None
        ) -> pd.DataFrame:
    """
    Selects only those rows from Dataframe which lie within given
    frequency range. If range is not given, returns original Dataframe.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for row filtration.

    Returns:
        pandas Dataframe with filtered rows by word frequency.
    """
    if freq_range is not None:
        ids = (freq_range[0] <= dataframe["frequency(ipm)"]) & \
            (dataframe["frequency(ipm)"] <= freq_range[1])
        return dataframe[ids]
    return dataframe


def load_and_prep_dataframe(
        pools_folder: str,
        initial_df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Creates single Dataframe, transforms labels into continuous and
    add information about word frequency and lemma.

    Args:
        pools_folder: directory with annotation results (tsv) from toloka.
        initial_df: tsv file with all sentences and their data.

    Returns:
        pandas Dataframe with words, continuous complexity, lemmas, metadata.
    """
    dataframe = merge_annotated_toloka_tsv(
        *[f for f in Path(pools_folder).rglob("*.tsv") if f.is_file()],
        drop_cols=[
            "GOLDEN:complexity",
            "HINT:text",
            "HINT:default_language",
            "ASSIGNMENT:assignment_id"
            ]
        )
    dataframe = project_labels_into_contunious(dataframe)
    dataframe = add_freq_for_sentence(
        dataframe,
        pd.read_csv(initial_df, sep="\t")
    )
    return dataframe


def filter_by_fast_responses(
        dataframe: pd.DataFrame,
        response_limit=15
        ) -> pd.DataFrame:
    """
    Filters Dataframe rows by selecting annotations submitted within
    a longer period of time that response_limit.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        response_limit: minimal allowed time in seconds for submitted task.

    Returns:
        pandas Dataframe without tasks completed too fast.
    """
    ids = ((dataframe["ASSIGNMENT:submitted"].dt.total_seconds()
           - dataframe["ASSIGNMENT:started"].dt.total_seconds())
           >= response_limit)
    return dataframe[ids]


def select_labels_for_common_lemmas(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame
        ) -> Tuple[pd.DataFrame]:
    """
    Selects complexity labels only for common lemmas in datasets.

    Args:
        l_dataframe: first Dataframe with complexity, contexts, and metadata.
        r_dataframe: second Dataframe with complexity, contexts, and metadata.

    Returns:
        two arrays with complexity labels of common lemmas from both datasets.
    """
    l_dataframe = aggregate_by_lemma(l_dataframe)
    r_dataframe = aggregate_by_lemma(r_dataframe)
    l_inter_ids = l_dataframe.index.isin(r_dataframe.index)
    r_inter_ids = r_dataframe.index.isin(l_dataframe.index)
    l_complexity = l_dataframe[l_inter_ids].sort_index()["OUTPUT:complexity"]
    r_complexity = r_dataframe[r_inter_ids].sort_index()["OUTPUT:complexity"]
    return l_complexity, r_complexity
