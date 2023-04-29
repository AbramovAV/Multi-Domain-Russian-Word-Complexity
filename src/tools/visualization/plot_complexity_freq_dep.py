from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import aggregate_by_lemma, \
    merge_annotated_toloka_tsv, add_freq_for_sentence, \
        project_labels_into_contunious, filter_by_freq_range, \
            filter_by_fast_responses
from src.tools.data_preparation.prepare_data_for_annotation import FREQUENCY_RANGES

sns.set_style("darkgrid")

def _annotate_points(dataframe: pd.DataFrame, ax:plt.Axes) -> plt.Axes:
    for task_id in range(0, dataframe.shape[0]):
        ax.text(dataframe["frequency(ipm)"][task_id]+0.01, dataframe["OUTPUT:complexity"][task_id], 
        dataframe["lemma"][task_id], horizontalalignment='left', 
        size='medium', color='black', weight='semibold')
    return ax


def plot_complexity_freq_dep(dataframe, decimate_ratio=0.01, save_dir='.',):
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize':(12, 8)})

    dataframe = aggregate_by_lemma(dataframe, auxiliary_mapping={"frequency(ipm)": "mean"})
    foreground_data = dataframe.sample(frac=decimate_ratio, axis='rows')
    background_data = dataframe[~dataframe.index.isin(foreground_data.index)]
    filename = "complexity_freq_dep.png"
    title = "Complexity-frequency dependency"
    ax1 = sns.scatterplot(
        x=foreground_data["frequency(ipm)"],
        y=foreground_data["OUTPUT:complexity"],
        alpha=1.0,
        )
    ax1 = _annotate_points(foreground_data, ax1)
    ax2 = sns.scatterplot(
        x=background_data["frequency(ipm)"],
        y=background_data["OUTPUT:complexity"],
        alpha=0.1,
        color='skyblue'
    )
    ax1.set_title(title)
    ax1.set_xlabel("Frequency(ipm)")
    ax1.set_ylabel("Word complexity score")
    ax1.set_ylim(0, 1)
    ax1.set_xscale("log")
    ax1.grid(True, which='major', axis='both')
    plt.savefig(str(Path(save_dir) / filename), bbox_inches='tight')
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--fast_responses_limit", default=15)
@click.option("--decimate_ratio", default=0.01)
@click.option("--save_dir", default=".")
def main(pools_folder, initial_df, fast_responses_limit, decimate_ratio, save_dir):
    dataframe = merge_annotated_toloka_tsv(
        *[f for f in Path(pools_folder).rglob("*.tsv") if f.is_file()],
        drop_cols=["GOLDEN:complexity",
                  "HINT:text",
                  "HINT:default_language",
                  "ASSIGNMENT:assignment_id"])
    dataframe = project_labels_into_contunious(dataframe)
    dataframe = add_freq_for_sentence(dataframe, pd.read_csv(initial_df, sep="\t"))
    plot_complexity_freq_dep(dataframe, decimate_ratio, save_dir=save_dir)

if __name__=='__main__':
    main()