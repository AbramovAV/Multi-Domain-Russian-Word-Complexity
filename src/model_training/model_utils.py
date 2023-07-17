"""
Module with utils for model loading
"""

from transformers import (BertForSequenceClassification, PreTrainedModel,
                          RobertaForSequenceClassification)


def get_model(name: str) -> PreTrainedModel:
    """
    Load pretrained model from Transformers repo.

    Args:
        name: model name as stated in Transformers repo.
              Available options: DeepPavlov/rubert-base-cased,
              sberbank-ai/ruRoberta-large

    Returns:
        pretrained PyTorch model
    """
    if name == "sberbank-ai/ruRoberta-large":
        model = RobertaForSequenceClassification.from_pretrained(
            name, num_labels=1)
    elif name == "DeepPavlov/rubert-base-cased":
        model = BertForSequenceClassification.from_pretrained(
            name, num_labels=1)
    else:
        raise ValueError(
            "Incorrect model name, available are:"
            "DeepPavlov/rubert-base-cased, sberbank-ai/ruRoberta-large")
    return model
