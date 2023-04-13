"""
Loads and preprocesses medline or UN corpus, samples target words
and creates tsv files for Toloka pools
"""
from __future__ import annotations

import random
from ast import literal_eval
from math import ceil
from pathlib import Path
from typing import Dict

import click
import numpy as np
import pandas as pd
import spacy
from spacy.lang.ru import Russian
from tqdm import tqdm

FREQUENCY_RANGES = (  # instances per million
    (51, 250),
    (251, 500),
    (501, 1400),
    (11, 50),
    (1401, 3100),
    (5, 10),
    (2, 4),
    (3101, 10000),
)  # changing order to inspect rare words first

MAX_SAMPLED_CONTEXTS_PER_LEMMA = 5
MIN_NUM_SAMPLED_CONTEXTS = 3100
MAX_NUM_SENTENCES_PER_TASK = 300
MIN_WORD_LEN = 4


def _load_medline_data(path: str) -> pd.DataFrame:
    """
    Loads data from tsv file with Medline corpus,
    splits RU part into sentences and returns DataFrame
    with separate sentences at each line.

    Args:
        path: path to tsv file with corpus

    Returns:
        pandas DataFrame with single column 'sentence'
    """
    nlp = Russian()
    nlp.add_pipe("sentencizer")
    dataframe = pd.read_csv(path, sep='\t')
    ru_texts = dataframe['ru']
    medline_sentences = {
        'sentence': []
    }
    for text in ru_texts:
        doc = nlp(text)
        for sentence in doc.sents:
            # splitting by newlines to find longest sentence in text
            max_sentence = max(sentence.text.strip().split("\n"))
            medline_sentences['sentence'].append(
                # replacing non-breaking spaces
                max_sentence.replace("\xa0", " ")
            )
    return pd.DataFrame.from_dict(medline_sentences)


def _load_un_data(path: str) -> pd.DataFrame:
    """
    Loads data from tsv file with UN corpus
    and returns new DataFrame with RU sentences only.

    Args:
        path: path to tsv file with corpus

    Returns:
        pandas DataFrame with single column 'sentence'
    """
    dataframe = pd.read_csv(path, sep='\t')
    un_sentences = {
        'sentence': dataframe['sentence']
    }
    return pd.DataFrame.from_dict(un_sentences)


def load_initial_data(path: str, source: str) -> pd.DataFrame:
    """
    Incapsulates loading funcs for medline and UN.

    Args:
        path: path to tsv file with data
        source: origins of data - medline or UN

    Returns:
        pandas DataFrame with single column 'sentence'
    """
    assert source in ['medline', 'un'], \
        "Currently you can choose only between Medline and UN corpora"
    func = f"_load_{source}_data(\"{path}\")"
    return literal_eval(func)


def load_frequency_dict(path: str) -> pd.DataFrame:
    """
    Loads frequency dictionary, downloaded from here:
    https://kpfu.ru/diahronicheskij-chastotnyj-slovar-russkoj-leksiki.html

    Args:
        path: path to csv file with dictionary

    Returns:
        Dataframe with words as indices and corresponding
        sentences and lemma's frequencies.
    """
    name_mapping = {
        "СЛОВО": "word",
        "РАНГ": "rank",
        "ОНТНОСИТЕЛЬНАЯ ЧАСТОТА (на 1 млн.)": "Freq(ipm)",
        "ДОКУМЕНТНАЯ ЧАСТОТА": "Doc",
        "ЛЕММЫ (по OpenCorpora)": "Lemmas",
        "POS (по GBN)": "PoS"
    }
    freq_dict = pd.read_csv(path, sep='\t').rename(columns=name_mapping)
    freq_dict = freq_dict.drop(labels=["rank", "Doc"], axis=1)
    freq_dict["PoS"] = freq_dict['PoS'].apply(
        lambda x: list(
                map(
                    lambda y: y.replace("%)", "").split("("),
                    x.replace(" ", "").split(",")
                ),
            )[0][0]
    )
    freq_dict["Lemmas"] = freq_dict["Lemmas"].apply(
        lambda x: list(filter(
            lambda w: w[1] in ['(NOUN)', '(NPRO)'],
            map(
                lambda y: y.split(" ")[1:],
                x.split(", "),
            ),
        ))
    )
    freq_dict = freq_dict[freq_dict["Lemmas"].map(len) > 0]
    freq_dict = freq_dict[freq_dict["PoS"] == "NOUN"].drop(columns=["PoS",])
    freq_dict = freq_dict.explode("Lemmas")
    freq_dict["Lemmas"] = freq_dict["Lemmas"].apply(
        lambda x: x[0]
    )
    freq_dict = freq_dict.groupby("Lemmas", sort=False).aggregate(
        {"word": set, "Freq(ipm)": "sum"}
    ).explode("word").reset_index().set_index("word")

    return freq_dict


def _prepare_data_to_sample_from(
        initial_data: pd.DataFrame,
        freq_dict: pd.DataFrame) -> pd.DataFrame:
    def add_data_to_sample_from(
                                data_to_sample: Dict,
                                word: str,
                                word_info: Dict,
                                sentence: str) -> Dict:
        if word_info["Lemmas"] not in data_to_sample:
            data_to_sample[word_info["Lemmas"]] = {
                'freq': word_info['Freq(ipm)'],
                'word_sntc_pairs': set(((word, sentence),)),
                }
        else:
            data_to_sample[word_info["Lemmas"]]['word_sntc_pairs'].add(
                (word, sentence)
            )
        return data_to_sample

    def apply_heuristics(word: str) -> bool:
        return len(word) >= MIN_WORD_LEN and \
            not word.isupper()

    data_to_sample = {}
    nlp = spacy.load('ru_core_news_lg')
    # in check are way faster for set() than for pandas Index
    freq_dict_index = set(freq_dict.index)
    freq_dict_dict = {
        k: g.to_dict(orient='records') for k, g in freq_dict.groupby(level=0)
        }
    for sentence in tqdm(initial_data['sentence']):
        doc = nlp(sentence)
        for token in doc:
            word = token.text if token.text in freq_dict_index \
                else token.text.lower()
            if word in freq_dict_index and \
                token.pos_ in ('NOUN', 'PROPN') and \
                    apply_heuristics(word):
                for word_info in freq_dict_dict[word]:
                    if word_info['Lemmas'] == token.lemma_:
                        break
                else:
                    continue
                data_to_sample = add_data_to_sample_from(
                    data_to_sample, token.text, word_info, sentence
                )
    return data_to_sample


def sample_data_for_annotation(
        initial_data: pd.DataFrame,
        freq_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Samples data to annotate with Toloka by collecting all
    suitable words for each available frequency range that
    appears in given frequency dictionary and randomly choosing
    among them until either available words or limit on maximum
    number of words per range is reached.

    Args:
        initial_data: dataframe with single column "sentence"
        freq_dict: dataframe with frequency dictionary with columns
        PoS and	Freq(ipm), and Lemma set as index.

    Returns:
        dataframe with sampled data, where each sample is described in 5
        columns: lemma, frequency(ipm), context, target_word, start_idx.
    """
    data_to_sample = _prepare_data_to_sample_from(initial_data, freq_dict)
    found_lemmas = list(data_to_sample.keys())
    random.shuffle(found_lemmas)
    resulting_data = {
        'lemma': [],
        'frequency(ipm)': [],
        'context': [],
        'target_word': [],
        'start_idx': []
    }

    freq_ranges_left = len(FREQUENCY_RANGES)
    contexts_to_sample = MIN_NUM_SAMPLED_CONTEXTS
    contexts_to_sample_per_range = int(contexts_to_sample / freq_ranges_left)
    for freq_range in FREQUENCY_RANGES[::-1]:
        contexts_left = contexts_to_sample_per_range
        for found_lemma in found_lemmas:
            random_lemma_data = data_to_sample.get(found_lemma, None)
            if random_lemma_data is not None:
                if freq_range[0] <= random_lemma_data['freq'] <= freq_range[1]:
                    pairs = list(random_lemma_data['word_sntc_pairs'])
                    random.shuffle(pairs)
                    for (word, context) in pairs[:5]:
                        resulting_data['lemma'].append(found_lemma)
                        resulting_data['frequency(ipm)'].append(
                            random_lemma_data['freq']
                        )
                        resulting_data['context'].append(context)
                        resulting_data['target_word'].append(word)
                        resulting_data['start_idx'].append(
                            context.index(word)
                        )
                        contexts_left -= 1
            if contexts_left <= 0:
                break
        contexts_to_sample -= (
            contexts_to_sample_per_range - max(0, contexts_left)
        )
        freq_ranges_left -= 1
        if contexts_left > 0:
            contexts_to_sample_per_range = ceil(
                contexts_to_sample / freq_ranges_left
            )

    return pd.DataFrame.from_dict(resulting_data)


def export_data_for_annotation(
        sampled_data: pd.DataFrame,
        path_to_save: str) -> None:
    """
    Exports sampled data formatted for Toloka, where target word
    in each sentence is highlighted as <mark>target_word</mark>.

    Args:
        sampled_data: dataframe with data to export
        path_to_save: folder to store prepared data
    """
    def add_marks(df_row: Dict) -> str:
        """
        Adds <mark> tags around target word
        """
        before_target = df_row['context'][:df_row['start_idx']]
        target = df_row['target_word']
        after_target = df_row['context'][df_row['start_idx'] + len(target):]
        return before_target + "<mark>" + target + "</mark>" + after_target

    path_to_save = Path(path_to_save)
    sampled_data = sampled_data.sample(frac=1)
    sub_dfs = np.array_split(
        sampled_data,
        ceil(len(sampled_data) / MAX_NUM_SENTENCES_PER_TASK)
    )

    for idx, sub_df in enumerate(sub_dfs):
        prepared_contexts = sub_df.apply(
            add_marks,
            axis=1
        )
        df_for_annotation = pd.DataFrame(prepared_contexts)
        df_for_annotation = df_for_annotation.reset_index(drop=True)
        df_for_annotation.columns = ["INPUT:text"]
        df_for_annotation.to_csv(
            str(path_to_save / f"pool_{idx+1}.tsv"),
            sep='\t',
            index=False
        )


@click.command()
@click.argument("save_dir")
@click.option("--data_source")
@click.option("--initial_data")
@click.option(
    "--freq_dict",
    default="data/dictionaries/rus_freq_dictionary_1992-2019.csv",
    help="path to frequency dictionary",
    show_default=True
)
@click.option(
    "--prepared_data",
    default=None,
    help="path to tsv file with pre-sampled \
        data before splitting into pool chunks",
    show_default=True
)
def main(save_dir: str,
         data_source: str,
         initial_data: str,
         freq_dict="data/dictionaries/rus_freq_dictionary_1992-2019.csv",
         prepared_data=None):
    """
    Samples words in different frequency ranges and prepares
    annotation files in Toloka format for several pools.

    DATA_SOURCE: source of data (i.e. un or medline)

    INITIAL_DATA: path to tsv file with data from data_source

    SAVE_DIR: path to folder where annotation files will be stored
    """
    if prepared_data is not None:
        sampled_data = pd.read_csv(prepared_data, sep='\t')
    else:
        initial_data = load_initial_data(initial_data, data_source)
        frequency_dict = load_frequency_dict(freq_dict)
        sampled_data = sample_data_for_annotation(initial_data, frequency_dict)
        path_to_save = Path(save_dir)
        path_to_save.mkdir(parents=True, exist_ok=True)
        sampled_data.to_csv(str(path_to_save / "data.tsv"), sep='\t')
    export_data_for_annotation(sampled_data, save_dir)


if __name__ == '__main__':
    main()  # pylint: disable=E1120