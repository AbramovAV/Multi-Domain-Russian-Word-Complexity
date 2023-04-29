from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tools.data_analysis.analysis_utils import aggregate_by_lemma, \
    merge_annotated_toloka_tsv, add_freq_for_sentence, \
        project_labels_into_contunious, filter_by_freq_range, \
            filter_by_fast_responses, load_and_prep_dataframe

sns.set_style("darkgrid")


def plot_intersected_complexity_dep(l_dataframe, r_dataframe, l_dataset_name="", r_dataset_name="", save_dir='.'):
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize':(12, 8)})

    l_dataframe = aggregate_by_lemma(l_dataframe)
    r_dataframe = aggregate_by_lemma(r_dataframe)
    l_inter_ids = l_dataframe.index.isin(r_dataframe.index)
    r_inter_ids = r_dataframe.index.isin(l_dataframe.index)
    l_complexity = l_dataframe[l_inter_ids].sort_index()["OUTPUT:complexity"]
    r_complexity = r_dataframe[r_inter_ids].sort_index()["OUTPUT:complexity"]

    A = np.vstack([l_complexity, np.ones(len(l_complexity))]).T
    slope, noise = np.linalg.lstsq(A, r_complexity, rcond=None)[0]

    ax = sns.scatterplot(x=l_complexity, y=r_complexity)
    sns.lineplot(x=l_complexity, y=slope * l_complexity + noise, ax=ax, color='orange', linewidth=2.5)
    ax.set_title(f"Dependency between complexity scores for datasets {l_dataset_name} and {r_dataset_name}")
    ax.set_xlabel(f"Complexity for dataset {l_dataset_name}")
    ax.set_ylabel(f"Complexity for dataset {r_dataset_name}")
    ax.set_xlim([0, l_complexity.max() + 0.05])
    ax.set_ylim([0, r_complexity.max() + 0.05])
    ax.grid(True, which='both', axis='both')
    plt.savefig(str(Path(save_dir) / f"complexity_dep_{l_dataset_name}_{r_dataset_name}.png"), bbox_inches='tight')
    plt.close()


@click.command()
@click.argument("pools_folder")
@click.argument("initial_df")
@click.argument("auxiliary_pools_folder")
@click.argument("auxiliary_initial_df")
@click.option("--main_dataset_name", default="")
@click.option("--aux_dataset_name", default="")
@click.option("--fast_responses_limit", default=15)
@click.option("--save_dir", default=".")
def main(pools_folder, initial_df, auxiliary_pools_folder, auxiliary_initial_df, main_dataset_name, aux_dataset_name, fast_responses_limit, save_dir):
    l_dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    r_dataframe = load_and_prep_dataframe(auxiliary_pools_folder, auxiliary_initial_df)
    plot_intersected_complexity_dep(l_dataframe, r_dataframe, main_dataset_name, aux_dataset_name, save_dir)


if __name__=='__main__':
    main()