"""
Main module for training the lexical complexity estimation model.
"""
from pathlib import Path

import click
from transformers import (Trainer,
                          TrainingArguments,
                          AutoModelForSequenceClassification)

from src.model_training.dataset_utils import (ComplexityDataset,
                                              get_tokenizer)
import pandas as pd


def run_inference(trainer: Trainer, test_dataset, save_dir, save_csv):
    predictions = trainer.predict(test_dataset)
    saved_res = {
        "contexts":[], "targets":[], "labels":[]
    }
    for idx, pred in enumerate(predictions.predictions):
        data = test_dataset.get_data(idx)
        saved_res["contexts"].append(data[0])
        saved_res["targets"].append(data[1])
        saved_res["labels"].append(pred.item())
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame.from_dict(saved_res).to_csv(Path(save_dir) / save_csv, index=False)


@click.command()
@click.option("--input_csv", default="./inputs.csv", show_default=True, help="path to csv file ")
@click.option(
    "--batch_size", default=128, show_default=True,
    help="Batch size for training data")
@click.option("--checkpoint", default="./checkpoint.pth", show_default=True)
@click.option("--save_dir", default="./results", show_default=True)
@click.option("--save_csv", default="predictions.csv", show_default=True)
def main(  # pylint: disable=too-many-arguments, too-many-locals
        input_csv="./inputs.csv",
        batch_size=128,
        checkpoint="./checkpoint.pth",
        save_dir="./results",
        save_csv="predictions.csv"
        ) -> None:
    """
    Script for running inference on data.
    """
    test_dataset = ComplexityDataset(
        pd.read_csv(input_csv).to_dict(orient="list"),
        get_tokenizer("DeepPavlov/rubert-base-cased"), has_labels=False)

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_eval_batch_size=batch_size,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        # checkpoint,
        "DeepPavlov/rubert-base-cased",
        from_tf=False,
        num_labels=1
    )
    trainer = Trainer(model, training_args)
    run_inference(trainer, test_dataset, save_dir, save_csv)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter