"""
Plots scatter plot to demonstrate a non-linear dependency between
a word complexity (y-axis) and word frequency (x-axis).
By default, reduces the number of elements with legend by a factor of 100.
"""
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import (aggregate_by_lemma,
                                                    load_and_prep_dataframe)

sns.set_style("darkgrid")


def _annotate_points(dataframe: pd.DataFrame, axis: plt.Axes) -> plt.Axes:
    """
    Writes a word corresponding to a specific point on scatter plot.

    Args:
        dataframe: pandas Dataframe with words freq, complexity and lemma.
        axis: pyplot axis object with scatter plot.

    Returns:
        pyplot axis object with scatter plot and some annotated points.
    """
    for task_id in range(0, dataframe.shape[0]):
        axis.text(
            dataframe["frequency(ipm)"][task_id]+0.01,
            dataframe["OUTPUT:complexity"][task_id],
            dataframe["lemma"][task_id],
            # pylint: disable=duplicate-code
            horizontalalignment="left",
            size="medium",
            color="black",
            weight="semibold"
            # pylint: disable=duplicate-code
        )
    return axis


def plot_complexity_freq_dep(
        dataframe: pd.DataFrame,
        decimate_ratio=0.01,
        save_dir='.'
        ) -> None:
    """
    Plots scatter plot with word complexity at y-axis and frequency at x-axis.

    Args:
        dataframe: pandas Dataframe with words freq, complexity and lemma.
        decimate_ratio: fraction of data to annotate.
        save_dir: directory to save an image with plot.

    Returns:
        None
    """
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    dataframe = aggregate_by_lemma(
        dataframe,
        auxiliary_mapping={"frequency(ipm)": "mean"}
    )
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
    _ = sns.scatterplot(
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
# @click.option("--fast_responses_limit", default=15)
@click.option("--decimate_ratio", default=0.01)
@click.option("--save_dir", default=".")
def main(
        pools_folder,
        initial_df,
        # fast_responses_limit,
        decimate_ratio,
        save_dir
        ) -> None:
    """
    POOLS_FOLDER: directory with annotation results (tsv) from toloka
    INITIAL_DF: tsv file with all sentences and their data (lemma, freq, word)
    """
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    plot_complexity_freq_dep(dataframe, decimate_ratio, save_dir=save_dir)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
