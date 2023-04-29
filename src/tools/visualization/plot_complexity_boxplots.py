from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import aggregate_by_lemma, load_and_prep_dataframe
from src.tools.data_preparation.prepare_data_for_annotation import FREQUENCY_RANGES


def plot_complexity_boxplots(dataframe, freq_ranges, save_dir='.'):
    def check_which_range_fits(freq):
        for freq_range in freq_ranges:
            if freq_range[0] <= freq <= freq_range[1]:
                return freq_range
    
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize':(12, 8)})

    dataframe["freq_range"] = dataframe["frequency(ipm)"].apply(check_which_range_fits)
    dataframe = aggregate_by_lemma(dataframe, auxiliary_mapping={"freq_range": "first"})
    ax = sns.boxplot(data=dataframe, x="freq_range", y="OUTPUT:complexity", orient='v', order=sorted(freq_ranges))
    ax.set_title("Boxplots for word complexity per frequency range (ipm)")
    ax.set_xlabel("Frequency range (ipm)")
    ax.set_ylabel("Complexity")
    ax.grid(True, which='both', axis='both')
    plt.savefig(str(Path(save_dir) / "boxplots.png"), bbox_inches='tight')
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.option("--fast_responses_limit", default=15)
@click.option("--save_dir", default=".")
def main(pools_folder, initial_df, fast_responses_limit=15, save_dir="."):
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    plot_complexity_boxplots(dataframe, sorted(FREQUENCY_RANGES), save_dir)


if __name__=="__main__":
    main()