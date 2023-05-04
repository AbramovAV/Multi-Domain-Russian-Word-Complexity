from pathlib import Path

import pandas as pd

from src.tools.data_preparation.prepare_data_for_annotation import _add_marks


def merge_annotated_toloka_tsv(*pathes, drop_cols=None):
    dataframe = pd.read_csv(pathes[0], sep='\t')
    for path in pathes[1:]:
        next_dataframe = pd.read_csv(path, sep='\t')
        dataframe = pd.concat([dataframe, next_dataframe], ignore_index=True)
    if drop_cols is not None:
        dataframe = dataframe.drop(drop_cols, axis="columns")
    return dataframe


def add_freq_for_sentence(dataframe:pd.DataFrame, initial_dataframe:pd.DataFrame):
    prep_contexts = initial_dataframe.apply(
            _add_marks,
            axis=1
    )
    initial_dataframe['context'] = prep_contexts
    dataframe = pd.merge(dataframe, initial_dataframe, how='left', 
                         left_on='INPUT:text', right_on='context')
    dataframe = dataframe.drop(labels=["start_idx", "context"], axis="columns")
    return dataframe


def project_labels_into_contunious(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe["OUTPUT:complexity"] = (dataframe["OUTPUT:complexity"] - 1) * 0.25
    return dataframe


def project_labels_into_discrete(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy(deep=True)
    dataframe["OUTPUT:complexity"] = ((dataframe["OUTPUT:complexity"] / 0.25) + 1).astype(int)
    return dataframe


def aggregate_by_task(dataframe:pd.DataFrame, freq_range=None, auxillary_mapping={}) -> pd.DataFrame:
    if freq_range is None:
        dataframe = dataframe.groupby("ASSIGNMENT:task_id", sort=False).aggregate(
        {"OUTPUT:complexity": "mean", "lemma": "first"} | auxillary_mapping
        )
    else:
        dataframe = dataframe.groupby("ASSIGNMENT:task_id", sort=False).aggregate(
        {"OUTPUT:complexity": "mean", "frequency(ipm)": "mean", "lemma": "first"} | \
            auxillary_mapping
        )
    return dataframe


def aggregate_by_lemma(dataframe:pd.DataFrame, freq_range=None, auxiliary_mapping={}) -> pd.DataFrame:
    if freq_range is None:
        dataframe = dataframe.groupby("lemma", sort=False).aggregate(
        {"OUTPUT:complexity": "mean", "lemma": "first"} | auxiliary_mapping
        )
    else:
        dataframe = dataframe.groupby("lemma", sort=False).aggregate(
        {"OUTPUT:complexity": "mean", "frequency(ipm)": "mean", "lemma": "first"} | \
            auxiliary_mapping
        )
    return dataframe

def filter_by_freq_range(dataframe:pd.DataFrame, freq_range=None):
    if freq_range is not None:
        ids = (freq_range[0] <= dataframe["frequency(ipm)"]) & \
            (dataframe["frequency(ipm)"] <= freq_range[1])
        return dataframe[ids]


def load_and_prep_dataframe(pools_folder:str, initial_df:pd.DataFrame) -> pd.DataFrame:
    dataframe = merge_annotated_toloka_tsv(
        *[f for f in Path(pools_folder).rglob("*.tsv") if f.is_file()],
        drop_cols=["GOLDEN:complexity",
                  "HINT:text",
                  "HINT:default_language",
                  "ASSIGNMENT:assignment_id"])
    dataframe = project_labels_into_contunious(dataframe)
    dataframe = add_freq_for_sentence(dataframe, pd.read_csv(initial_df, sep="\t"))
    return dataframe


def filter_by_fast_responses(dataframe: pd.DataFrame, response_limit=15) -> pd.DataFrame:
    ids = (dataframe["ASSIGNMENT:submitted"].dt.total_seconds() - \
        dataframe["ASSIGNMENT:started"].dt.total_seconds()) >= response_limit
    return dataframe[ids]


if __name__=='__main__':
    dataframe = pd.read_csv("data/annotated/annotated_data/medline/assignments_from_pool_38509153__11-04-2023.tsv", sep='\t')
    initial_df = pd.read_csv("data/annotated/prepared_data/medline/data.tsv", sep='\t')
    add_freq_for_sentence(dataframe, initial_df)