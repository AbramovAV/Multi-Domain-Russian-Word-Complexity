"""
Plots scatter plot with dependency between word complexity aggregated by lemma
for different datasets and its linear approximation.
"""
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import (load_and_prep_dataframe,
                                                    select_common_lemmas)

sns.set_style("darkgrid")


def _annotate_points(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        axis: plt.Axes
        ) -> plt.Axes:
    """
    Writes a word corresponding to a specific point on scatter plot.

    Args:
        dataframe: pandas Dataframe with words freq, complexity and lemma.
        axis: pyplot axis object with scatter plot.

    Returns:
        pyplot axis object with scatter plot and some annotated points.
    """
    for lemma in l_dataframe.index:
        axis.text(
            l_dataframe.loc[lemma]["OUTPUT:complexity"],
            r_dataframe.loc[lemma]["OUTPUT:complexity"],
            lemma,
            # pylint: disable=duplicate-code
            horizontalalignment="left",
            size="medium",
            color="black",
            weight="semibold"
            # pylint: disable=duplicate-code
        )
    return axis


def plot_intersected_complexity_dep(
        l_dataframe: pd.DataFrame,
        r_dataframe: pd.DataFrame,
        l_dataset_name="",
        r_dataset_name="",
        save_dir='.') -> None:
    """
    Plots scatter plot with word complexities from one dataset on x-axis
    and word complexities from another dataset on y-axis.

    Args:
        l_dataframe: x-axis Dataframe with complexity, contexts, and metadata.
        r_dataframe: y-axis Dataframe with complexity, contexts, and metadata.
        l_dataset_name: source of origin of x-axis dataset.
        r_dataset_name: source of origin of y-axis dataset.
        save_dir: directory to save image with plot.

    Returns:
        None
    """
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize': (12, 8)})
    l_dataframe, r_dataframe = select_common_lemmas(
        l_dataframe, r_dataframe)
    l_score = l_dataframe["OUTPUT:complexity"]
    r_score = r_dataframe["OUTPUT:complexity"]
    slope, noise = np.linalg.lstsq(
        np.vstack([l_score, np.ones(len(l_score))]).T,
        r_score,
        rcond=None
    )[0]
    axis = sns.scatterplot(x=l_score, y=r_score)
    sns.lineplot(
        x=l_score,
        y=slope * l_score + noise,
        ax=axis,
        color="orange",
        linewidth=2.5)
    axis.set_title(
        "Dependency between complexity scores for datasets "
        f"{l_dataset_name} and {r_dataset_name}")
    axis.set_xlabel(f"Complexity for dataset {l_dataset_name}")
    axis.set_ylabel(f"Complexity for dataset {r_dataset_name}")
    axis.set_xlim([0, l_score.max() + 0.05])
    axis.set_ylim([0, r_score.max() + 0.05])
    axis.grid(True, which='both', axis='both')
    outliers_condition = (l_score >= 0.3) | (r_score >= 0.3)
    axis = _annotate_points(
        l_dataframe[outliers_condition],
        r_dataframe[outliers_condition],
        axis
    )
    filename = f"complexity_dep_{l_dataset_name}_{r_dataset_name}.png"
    plt.savefig(str(Path(save_dir) / filename), bbox_inches='tight')
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.argument("auxiliary_pools_folder")
@click.argument("auxiliary_initial_df")
@click.option("--main_dataset_name", default="")
@click.option("--aux_dataset_name", default="")
# @click.option("--fast_responses_limit", default=15)
@click.option("--save_dir", default=".")
def main(  # pylint: disable=too-many-arguments
        pools_folder,
        initial_df,
        auxiliary_pools_folder,
        auxiliary_initial_df,
        main_dataset_name,
        aux_dataset_name,
        # fast_responses_limit,
        save_dir
        ) -> None:
    """
    POOLS_FOLDER: directory with annotation results (tsv) from toloka
    INITIAL_DF: tsv file with all sentences and their data (lemma, freq, word)
    """
    l_dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    r_dataframe = load_and_prep_dataframe(
        auxiliary_pools_folder, auxiliary_initial_df)
    plot_intersected_complexity_dep(
        l_dataframe, r_dataframe, main_dataset_name,
        aux_dataset_name, save_dir)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
