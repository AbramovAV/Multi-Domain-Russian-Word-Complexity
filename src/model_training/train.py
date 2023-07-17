"""
Main module for training the lexical complexity estimation model.
"""
import click
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments

from src.model_training.dataset_utils import (ComplexityDataset,
                                              compute_metrics, get_tokenizer,
                                              prepare_dataset)
from src.model_training.model_utils import get_model


def run_experiment(
        model_name, train_dataset, test_dataset,
        training_args, k_folds=10
        ) -> None:
    """
    Runs either single experiment or series of experiment for
    cross-validation.

    Args:
        model_name: model name to load pretrained model from Transformers repo
        train_dataset: ComplexityDataset instance with training data
        test_dataset: ComplexityDataset instance with test data
        training_args: arguments for Trainer,
        e.g. learning rate, weight decay, etc.
        k_folds: number of folds for cross-validation
    """
    if test_dataset is not None:
        model = get_model(model_name)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        result = trainer.evaluate()
        print(result)
    else:
        folds = KFold(n_splits=k_folds, shuffle=True)
        for train_indices, test_indices in folds.split(train_dataset):
            train_subset = train_dataset.from_indices(train_indices)
            test_subset = train_dataset.from_indices(test_indices)

            model = get_model(model_name)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=test_subset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            result = trainer.evaluate()
            print(result)


@click.command()
@click.argument("train_dataset_folder")
@click.argument("train_initial_data")
@click.argument("test_dataset_folder")
@click.argument("test_initial_data")
@click.option(
    "--k_folds", default=5, show_default=True,
    help="Number of CV-folds to validate model if train and test match")
@click.option(
    "--train_epochs", default=200, show_default=True,
    help="Number of training epochs")
@click.option(
    "--batch_size", default=128, show_default=True,
    help="Batch size for training data")
@click.option(
    "--warmup_steps", default=5, show_default=True,
    help="Number of warmup epochs before model training")
@click.option("--learning_rate", default=1e-5, show_default=True)
@click.option("--weight_decay", default=0.01, show_default=True)
@click.option("--logging_dir", default="./logs", show_default=True)
@click.option("--logging_steps", default=50, show_default=True)
@click.option("--save_steps", default=500, show_default=True)
@click.option("--evaluation_strategy", default="steps", show_default=True)
def main(  # pylint: disable=too-many-arguments, too-many-locals
        train_dataset_folder,
        train_initial_data,
        test_dataset_folder,
        test_initial_data,
        k_folds=5,
        train_epochs=50,
        batch_size=128,
        warmup_steps=5,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps",
        ) -> None:
    """
    Main functions for conducting experiments
    """
    train_dataset = ComplexityDataset(
        prepare_dataset(train_dataset_folder, train_initial_data),
        get_tokenizer("DeepPavlov/rubert-base-cased")
        )
    if test_dataset_folder != train_dataset_folder:
        test_dataset = ComplexityDataset(
            prepare_dataset(test_dataset_folder, test_initial_data),
            get_tokenizer("DeepPavlov/rubert-base-cased")
            )
    else:
        test_dataset = None

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy=evaluation_strategy,
    )

    run_experiment(
            "DeepPavlov/rubert-base-cased",
            train_dataset,
            test_dataset,
            k_folds=k_folds,
            training_args=training_args
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
