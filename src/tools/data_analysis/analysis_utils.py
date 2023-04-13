from src.tools.data_preparation.prepare_data_for_annotation import _add_marks
import pandas as pd

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

if __name__=='__main__':
    dataframe = pd.read_csv("data/annotated/annotated_data/medline/assignments_from_pool_38509153__11-04-2023.tsv", sep='\t')
    initial_df = pd.read_csv("data/annotated/prepared_data/medline/data.tsv", sep='\t')
    add_freq_for_sentence(dataframe, initial_df)