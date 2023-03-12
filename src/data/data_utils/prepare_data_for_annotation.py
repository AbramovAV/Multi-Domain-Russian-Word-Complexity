from __future__ import annotations
from math import ceil
from pathlib import Path
import random

import numpy as np
import pandas as pd
import spacy
from spacy.lang.ru import Russian
from tqdm import tqdm

FREQUENCY_RANGES = ( #instances per million
    (51,250),
    (251,500),
    (501,1400),
    (11,50),
    (1401,3100),
    (5,10),
    (2,4), 
    (3101,10000),
) # changing order to inspect rare words first

MAX_SAMPLED_CONTEXTS_PER_LEMMA = 5
MIN_NUM_SAMPLED_CONTEXTS = 3000
MAX_NUM_SENTENCES_PER_TASK = 300

def _load_medline_data(path:str) -> pd.DataFrame:
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
    df = pd.read_csv(path, sep='\t')
    ru_texts = df['ru']
    medline_sentences = {
        'sentence': []
    }
    for text in ru_texts:
        doc = nlp(text)
        for sentence in doc.sents:
            max_sentence = max(sentence.text.strip().split("\n")) # looking for the longest part in sentence, since some contain newline symbols
            medline_sentences['sentence'].append(
                max_sentence.replace(u"\xa0", u" ") # replacing non-breaking spaces with regular ones
            )
    return pd.DataFrame.from_dict(medline_sentences)


def _load_un_data(path:str) -> pd.DataFrame:
    """
    Loads data from tsv file with UN corpus
    and returns new DataFrame with RU sentences only.

    Args:
        path: path to tsv file with corpus

    Returns:
        pandas DataFrame with single column 'sentence'
    """
    df = pd.read_csv(path, sep='\t')
    un_sentences = {
        'sentence': df['sentence']
    }
    return pd.DataFrame.from_dict(un_sentences)

def load_initial_data(path:str, source:str) -> pd.DataFrame:
    assert source in ['medline', 'un'], "Currently you can choose only between Medline and UN corpora"
    func = f"_load_{source}_data(\"{path}\")"
    return eval(func)

def load_frequency_dict(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df = df.set_index("Lemma")
    return df

def sample_data_for_annotation(initial_data:pd.DataFrame, freq_dict:pd.DataFrame) -> pd.DataFrame:
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
    def add_data_to_sample_from(lemma_info):
        if lemma_info['PoS'] == 's' and \
            token.pos_ == 'NOUN' or \
                lemma_info['PoS'] == 's.PROP' and \
                    token.pos_ == 'PROPN':
            if lemma not in data_to_sample:
                data_to_sample[lemma] = {
                    'freq': lemma_info['Freq(ipm)'],
                    'contexts': set(((token.text, sentence),)),
                    }
            else:
                data_to_sample[lemma]['contexts'].add((token.text, sentence))

    data_to_sample = {}
    nlp = spacy.load('ru_core_news_lg')
    for sentence in tqdm(initial_data['sentence']):
        doc = nlp(sentence)
        for token in doc:
            lemma = token.lemma_ if token.lemma_ in freq_dict.index \
                 else token.lemma_.capitalize()
            if lemma in freq_dict.index:
                lemmas_info = freq_dict.loc[lemma]
                if isinstance(lemmas_info, pd.DataFrame):
                    for idx in range(len(lemmas_info)):
                        lemma_info = lemmas_info.iloc[idx]
                        add_data_to_sample_from(lemma_info)
                else:
                    add_data_to_sample_from(lemmas_info)

    found_lemmas = list(data_to_sample.keys())
    random.shuffle(found_lemmas)
    resulting_data = {
        'lemma':[],
        'frequency(ipm)':[],
        'context':[],
        'target_word':[],
        'start_idx': []
    }

    freq_ranges_left = len(FREQUENCY_RANGES)
    contexts_to_sample = MIN_NUM_SAMPLED_CONTEXTS
    contexts_to_sample_per_range = int(contexts_to_sample / freq_ranges_left) # per freq range
    for FREQ_RANGE in FREQUENCY_RANGES[::-1]:
        contexts_left = contexts_to_sample_per_range
        for found_lemma in found_lemmas:
            random_lemma_data = data_to_sample.get(found_lemma, None)
            if random_lemma_data is not None:
                freq = random_lemma_data['freq']
                if FREQ_RANGE[0] <= freq <= FREQ_RANGE[1]:
                    contexts = list(random_lemma_data['contexts'])
                    random.shuffle(contexts)
                    contexts = contexts[:5]
                    for context in contexts:
                        resulting_data['lemma'].append(found_lemma)
                        resulting_data['frequency(ipm)'].append(freq)
                        resulting_data['context'].append(context[1])
                        resulting_data['target_word'].append(context[0])
                        resulting_data['start_idx'].append(context[1].index(context[0]))
                        contexts_left -= 1
            if contexts_left<=0:
                break
        contexts_to_sample -= (contexts_to_sample_per_range - max(0, contexts_left))
        freq_ranges_left -= 1
        if contexts_left > 0:
            contexts_to_sample_per_range = ceil(contexts_to_sample / freq_ranges_left)

    return pd.DataFrame.from_dict(resulting_data)        


def export_data_for_annotation(sampled_data:pd.DataFrame, path_to_save:str) -> None:
    """
    Exports sampled data formatted for Toloka, where target word
    in each sentence is highlighted as <mark>target_word</mark>.

    Args:
        sampled_data: dataframe with data to export
        path_to_save: folder to store prepared data
    """
    path_to_save = Path(path_to_save)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    sampled_data.to_csv(str(path_to_save / "data.tsv"), sep='\t')
    sampled_data = sampled_data.sample(frac=1)
    sub_dfs = np.array_split(sampled_data, ceil(len(sampled_data) / MAX_NUM_SENTENCES_PER_TASK))
    for idx, sub_df in enumerate(sub_dfs):
        prepared_contexts = sub_df.apply(
            lambda x: x['context'][:x['start_idx']] + \
                      "<mark>" + \
                      x['context'][x['start_idx']:x['start_idx'] + len(x['target_word'])] + \
                      "</mark>" + \
                      x['context'][x['start_idx'] + len(x['target_word']):],
            axis=1
        )
        df_for_annotation = pd.DataFrame(prepared_contexts).reset_index(drop=True)
        df_for_annotation.columns = ["INPUT:text"]
        df_for_annotation.to_csv(str(path_to_save / f"pool_{idx+1}.tsv"), sep='\t')


if __name__=='__main__':
    initial_data = load_initial_data("data/initial/un_2014_ru.tsv", "un")
    frequency_dict = load_frequency_dict("data/dictionaries/freqrnc2011.csv")
    sampled_data = sample_data_for_annotation(initial_data, frequency_dict)
    export_data_for_annotation(sampled_data, "data/annotated/prepared_data/un")
