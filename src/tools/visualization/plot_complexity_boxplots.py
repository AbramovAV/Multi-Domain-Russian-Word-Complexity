"""
Plots boxplots for each frequency range to illustate word complexity stats.
"""
from pathlib import Path
from typing import Tuple

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import (aggregate_by_lemma,
                                                    load_and_prep_dataframe)
from src.tools.data_preparation import FREQUENCY_RANGES


def plot_complexity_boxplots(
        dataframe: pd.DataFrame,
        freq_ranges: Tuple[Tuple[int]],
        save_dir='.'
        ) -> None:
    """
    Plots word complexity boxplots computed per each frequency range.

    Args:
        dataframe: pandas Dataframe with complexity, contexts, and metadata.
        freq_ranges: frequency ranges used for selecting words for boxplot.
        save_dir: directory to save image with plot.

    Returns:
        None
    """
    def check_which_range_fits(freq: float) -> Tuple[int]:
        """
        Finds within which frequency range lies given frequency value.

        Args:
            freq: single word frequency value.

        Returns:
            Corresponding frequency range.
        """
        for freq_range in freq_ranges:
            if freq_range[0] <= freq <= freq_range[1]:
                return freq_range
        return None

    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    dataframe["freq_range"] = dataframe["frequency(ipm)"].apply(
        check_which_range_fits
    )
    dataframe = aggregate_by_lemma(
        dataframe, auxiliary_mapping={"freq_range": "first"}
    )
    axis = sns.boxplot(
        data=dataframe,
        x="freq_range",
        y="OUTPUT:complexity",
        orient='v',
        order=sorted(freq_ranges)
    )
    axis.set_title("Boxplots for word complexity per frequency range (ipm)")
    axis.set_xlabel("Frequency range (ipm)")
    axis.set_ylabel("Complexity")
    axis.grid(True, which='both', axis='both')
    plt.savefig(str(Path(save_dir) / "boxplots.png"), bbox_inches='tight')
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
# @click.option("--fast_responses_limit", default=15)
@click.option("--save_dir", default=".")
def main(
        pools_folder,
        initial_df,
        # fast_responses_limit=15,
        save_dir="."
        ) -> None:
    """
    POOLS_FOLDER: directory with annotation results (tsv) from toloka
    INITIAL_DF: tsv file with all sentences and their data (lemma, freq, word)
    """
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    plot_complexity_boxplots(dataframe, sorted(FREQUENCY_RANGES), save_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
