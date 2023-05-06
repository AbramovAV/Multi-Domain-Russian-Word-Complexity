"""
Plots histogram to illustrate word complexity distribution.
Optionally, does that for every frequency range among those used for sampling.
"""
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import (aggregate_by_lemma,
                                                    filter_by_freq_range,
                                                    load_and_prep_dataframe)
from src.tools.data_preparation import FREQUENCY_RANGES


def plot_dist(dataframe: pd.DataFrame, freq_range=None, save_dir=".") -> None:
    """
    Plots word complexity distribution aggregated by lemma as histogram
    and, optionally, does that only for given frequency range.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_range: frequency range for aggregation.
        save_dir: directory to save image with plot.

    Returns:
        None
    """
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    dataframe = aggregate_by_lemma(dataframe, freq_range)
    if freq_range is None:
        filename = "complexity_scores_histplot.png"
        title = "Complexity scores distribution"
    else:
        dataframe = filter_by_freq_range(dataframe, freq_range)
        filename = f"complexity_scores_histplot_{freq_range}(ipm).png"
        title = "Complexity scores distribution for frequency range" + \
            f"{freq_range[0]}-{freq_range[1]}"
    axis = sns.histplot(dataframe["OUTPUT:complexity"])
    axis.set_title(title)
    axis.set_xlabel("Word complexity score")
    axis.set_ylabel("Frequency")
    axis.set_xlim(0, 1)
    axis.grid(True, which='both', axis='both')
    plt.savefig(str(Path(save_dir) / filename))
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--split_by_freq_ranges", is_flag=True)
# @click.option("--fast_responses_limit", default=15)
@click.option("--save_dir", default=".")
def main(
        pools_folder,
        split_by_freq_ranges,
        initial_df,
        # fast_responses_limit,
        save_dir
        ) -> None:
    """
    POOLS_FOLDER: directory with annotation results (tsv) from toloka
    INITIAL_DF: tsv file with all sentences and their data (lemma, freq, word)
    """
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    if split_by_freq_ranges:
        for freq_range in sorted(FREQUENCY_RANGES):
            plot_dist(dataframe, freq_range, save_dir=save_dir)
    else:
        plot_dist(dataframe, None, save_dir=save_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
