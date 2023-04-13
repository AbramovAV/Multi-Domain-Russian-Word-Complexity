from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.tools.data_analysis.analysis_utils import merge_annotated_toloka_tsv, add_freq_for_sentence
from src.tools.data_preparation.prepare_data_for_annotation import FREQUENCY_RANGES


def compute_mean_complexity(dataframe:pd.DataFrame, freq_range=None) -> float:
    if freq_range is None:
        return dataframe["OUTPUT:complexity"].mean()
    else:
        ids = (freq_range[0] <= dataframe["frequency(ipm)"]) & \
            (dataframe["frequency(ipm)"] <= freq_range[1])
        return dataframe[ids]["OUTPUT:complexity"].mean()


def compute_std_complexity(dataframe:pd.DataFrame, freq_range=None):
    if freq_range is None:
        return dataframe["OUTPUT:complexity"].std()
    else:
        ids = (freq_range[0] <= dataframe["frequency(ipm)"]) & \
            (dataframe["frequency(ipm)"] <= freq_range[1])
        return dataframe[ids]["OUTPUT:complexity"].std()


def compute_annotator_agreement(dataframe, freq_range=None):
    pass


@click.command()
@click.argument("pools_folder")
@click.option("--split_by_freq_ranges", is_flag=True)
@click.option("--initial_df", default=None)
@click.option("--fast_responses_limit", default=15)
def main(pools_folder, split_by_freq_ranges, initial_df, fast_responses_limit):
    dataframe = merge_annotated_toloka_tsv(
        *[f for f in Path(pools_folder).rglob("*.tsv") if f.is_file()],
        drop_cols=["GOLDEN:complexity",
                  "HINT:text",
                  "HINT:default_language",
                  "ASSIGNMENT:assignment_id"])
    if split_by_freq_ranges:
        dataframe = add_freq_for_sentence(dataframe, pd.read_csv(initial_df, sep="\t"))
        for freq_range in sorted(FREQUENCY_RANGES):
            mean = compute_mean_complexity(dataframe, freq_range)
            std = compute_std_complexity(dataframe, freq_range)
            print(f"Mean and std complexity for for freq range {freq_range[0]}-{freq_range[1]} ipm:")
            print(np.round(mean, 3), np.round(std, 3))
            # compute_annotator_agreement(dataframe, freq_range)
    else:
        mean = compute_mean_complexity(dataframe)
        std = compute_std_complexity(dataframe)
        # compute_annotator_agreement(dataframe)
        print(f"Overall mean complexity: {np.round(mean, 3)}")
        print(f"Overall std complexity: {np.round(std, 3)}")

if __name__ == '__main__':
    main()

