from transformers import (BertForSequenceClassification,
                          RobertaForSequenceClassification)


def get_model(name: str):
    if name == "sberbank-ai/ruRoberta-large":
        model = RobertaForSequenceClassification.from_pretrained(
            name, num_labels=1)
    elif name == "DeepPavlov/rubert-base-cased":
        model = BertForSequenceClassification.from_pretrained(
            name, num_labels=1)
    else:
        raise ValueError("Incorrect model name, available are: DeepPavlov/rubert-base-cased, sberbank-ai/ruRoberta-large")
    return model
