"""
Module with various utils for processing data
"""
from typing import Dict, List

import numpy as np
import torch
from pandas import DataFrame, Series
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.tools.data_analysis.analysis_utils import (aggregate_by_task,
                                                    load_and_prep_dataframe)


class ComplexityDataset(Dataset):
    """
    Base dataset class for creating pairs contexts - complexity score.

    Args:
        data: dictionary with Series with contexts, scores and target words.
        tokenizer: pretrained model's tokenizer from Transformers repo.
    """
    def __init__(
            self, data: Dict[str, Series], tokenizer: PreTrainedTokenizer, has_labels=True
            ) -> None:
        if has_labels:
            self.keys = ["contexts", "scores", "targets"]
        else:
            self.keys = ["contexts", "targets"]
        self.has_labels = has_labels
        self.data = list(zip(*[data[key] for key in self.keys]))
        self.tokenizer = tokenizer
        self.tokenized_data = self._tokenize_and_align_labels()

    def _tokenize_and_align_labels(self) -> List:
        """
        Tokenizes data with self.tokenizer and packs it
        with complexity scores and attention masks.
        """
        tokenized_data = []
        for sample in self.data:
            try:
                if self.has_labels:
                    context, score, target = sample
                else:
                    context, target = sample
            except ValueError:
                raise ValueError(f"Mismatch between data in sample and value of has_labels. Make sure to pass correct data.")
            tokenized_inputs = self.tokenizer(
                text=context, text_pair=target, truncation=True,
                padding='max_length', return_tensors='pt', max_length=48)
            if self.has_labels:
                tokenized_inputs["labels"] = np.array(score).astype(np.float32)
            tokenized_inputs['attention_mask'] = torch.squeeze(
                tokenized_inputs['attention_mask']).tolist()
            tokenized_inputs['input_ids'] = torch.squeeze(
                tokenized_inputs['input_ids']).tolist()
            tokenized_inputs['token_type_ids'] = torch.squeeze(
                tokenized_inputs['token_type_ids']).tolist()
            tokenized_data.append(tokenized_inputs)
        return tokenized_data

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def get_data(self, idx):
        """
        Returns sample of original data (not tokenized).
        """
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def from_indices(self, indices: List[int]) -> "ComplexityDataset":
        """
        Creates new Dataset class with data sampled
        by given indices.

        Args:
            indices: indices of data samples to include in new class.

        Returns:
            new instance of ComplexityDataset
        """
        data = {key:[] for key in self.keys}
        for idx in indices:
            for key_idx, key in self.keys:
                data[key].append(self.data[idx][key_idx])
        return ComplexityDataset(data, self.tokenizer)


def get_tokenizer(name: str) -> PreTrainedTokenizer:
    """
    Loads pretrained Tokenizer from Transformers repo.
    """
    return AutoTokenizer.from_pretrained(name)


def prepare_dataset(
        pools_folder: str,
        initial_df: DataFrame
        ) -> Dict[str, Series]:
    """
    Preprocesses original dataset to use in training
    in the future.

    Args:
        pools_folder: directory with annotation results (tsv) from toloka.
        initial_df: tsv file with all sentences and their data.
    """
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    dataframe = aggregate_by_task(dataframe, auxiliary_mapping={
        "target_word": "first",
        "INPUT:text": "first"
    })
    targets = dataframe["target_word"]
    contexts = dataframe["INPUT:text"].map(
        lambda x: x.replace("<mark>", "").replace("</mark>", ""))
    scores = dataframe["OUTPUT:complexity"]
    dataset = {
        "contexts": contexts,
        "scores": scores,
        "targets": targets
    }
    return dataset


def run_on_validation_set(model, test_dataset):
    """
    Runs model on test dataset and print predictions.
    """
    for inputs in test_dataset:
        inputs['input_ids'] = torch.unsqueeze(
            inputs['input_ids'], 0).to("cuda")
        inputs['attention_mask'] = torch.unsqueeze(
            inputs['attention_mask'], 0).to("cuda")
        result = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'])
        print(
            result.logits.item(), inputs['labels'].item(),
            inputs['sentence']
        )


def compute_metrics(pred, which="both"):
    """
    Computes MAE and Pearson Correlation Coefficient
    on given predictions and ground truth data.
    """
    labels = list(pred.label_ids)
    preds = [x[0] for x in pred.predictions]
    mae = mean_absolute_error(labels, preds)
    corr, _ = pearsonr(labels, preds)
    if which == "both":
        metrics = {
            'mean_absolute_error': mae,
            'correlation_coefficient': corr
        }
    elif which == "mae":
        metrics = {'mean_absolute_error': mae}
    elif which == "corr":
        metrics = {'correlation_coefficient': corr}

    return metrics
