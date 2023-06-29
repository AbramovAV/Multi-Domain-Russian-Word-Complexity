import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.tools.data_analysis.analysis_utils import (aggregate_by_task,
                                                    load_and_prep_dataframe)


class ComplexityDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = [
            (cntx, cmplx, trgt) for (cntx, cmplx, trgt) in zip(data["contexts"], data["scores"], data["targets"])]
        self.tokenizer = tokenizer
        self.tokenized_data = self._tokenize_and_align_labels()

    def _tokenize_and_align_labels(self):
        tokenized_data = []
        for context, score, target in self.data:
            tokenized_inputs = self.tokenizer(text=context, text_pair=target, truncation=True, padding='max_length', return_tensors='pt', max_length=48)
            tokenized_inputs["labels"] = np.array(score)
            tokenized_inputs["labels"] = tokenized_inputs["labels"].astype(np.float32)
            tokenized_inputs['attention_mask'] = torch.squeeze(tokenized_inputs['attention_mask']).tolist()
            tokenized_inputs['input_ids'] = torch.squeeze(tokenized_inputs['input_ids']).tolist()
            tokenized_inputs['token_type_ids'] = torch.squeeze(tokenized_inputs['token_type_ids']).tolist()
            tokenized_data.append(tokenized_inputs)
        return tokenized_data

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def get_data(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def from_indices(self, indices):
        data = {
            "contexts": [],
            "scores": [],
            "targets": []
        }
        for idx in indices:
            data["contexts"].append(self.data[idx][0])
            data["scores"].append(self.data[idx][1])
            data["targets"].append(self.data[idx][2])
        return ComplexityDataset(data, self.tokenizer)


def get_tokenizer(name: str):
  return AutoTokenizer.from_pretrained(name)


def prepare_dataset(pools_folder, initial_df):
    dataframe = load_and_prep_dataframe(pools_folder, initial_df)
    dataframe = aggregate_by_task(dataframe, auxiliary_mapping={
        "target_word": "first",
        "INPUT:text": "first"
    })
    targets = dataframe["target_word"]
    contexts = dataframe["INPUT:text"].map(
        lambda x: x.replace("<mark>","").replace("</mark>",""))
    scores = dataframe["OUTPUT:complexity"]
    dataset = {
        "contexts": contexts,
        "scores": scores,
        "targets": targets
    }
    return dataset


def run_on_validation_set(model, test_dataset):
    for inputs in test_dataset:
        inputs['input_ids'] = torch.unsqueeze(inputs['input_ids'], 0).to("cuda")
        inputs['attention_mask'] = torch.unsqueeze(inputs['attention_mask'], 0).to("cuda")
        # inputs['token_type_ids'] = torch.unsqueeze(inputs['token_type_ids'], 0).to("cuda")

        # result = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
        result = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        print(result.logits.item(), inputs['labels'].item(), inputs['sentence'])


def compute_metrics(pred):
    labels = [x for x in pred.label_ids]
    preds = [x[0] for x in pred.predictions]
    # calculate accuracy using sklearn's function
    # print(f"labels : {labels[:10]}")
    # print(f"pred : {preds[:50]}")
    mae = mean_absolute_error(labels, preds)
    mape = mean_absolute_percentage_error(labels, preds)
    # correlation = np.corrcoef([labels, preds])[0, 1]
    corr, pval = pearsonr(labels, preds)
    return {
        'mean_absolute_error': mae,
        'mean_absolute_percentage_error': mape,
        'correlation_coefficient': corr
    }
