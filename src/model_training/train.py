import click
import numpy as np
from dataset_utils import (ComplexityDataset, compute_metrics, get_tokenizer,
                           prepare_dataset, run_on_validation_set)
from model_utils import get_model
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments


def run_experiment(model, train_dataset, test_dataset, training_args, k_folds=10):
    if test_dataset is not None:
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
        kf = KFold(n_splits=k_folds, shuffle=True)
        maes = []
        corrs = []
        for train_indices, test_indices in kf.split(train_dataset):
            train_subset = train_dataset.from_indices(train_indices)
            test_subset = train_dataset.from_indices(test_indices)

            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=test_subset,
            compute_metrics=compute_metrics,
            )

            trainer.train()
            result = trainer.evaluate()
            maes.append(result["mean_absolute_error"])
            corrs.append(result["correlation_coefficient"])
            print(result)
        print(f"Mean MAE: {np.mean(maes)}, mean corr: {np.mean(corrs)}")

@click.command()
@click.argument("train_dataset_folder")
@click.argument("train_initial_data")
@click.argument("test_dataset_folder")
@click.argument("test_initial_data")
@click.option("--k_folds", default=5, show_default=True, help="Number of CV-folds to validate model if train and test match")
@click.option("--train_epochs", default=200, show_default=True, help="Number of training epochs")
@click.option("--batch_size", default=128, show_default=True, help="Batch size for training data")
@click.option("--warmup_steps", default=5, show_default=True, help="Number of warmup epochs before model training")
@click.option("--learning_rate", default=1e-5, show_default=True)
@click.option("--weight_decay", default=0.01, show_default=True)
@click.option("--logging_dir", default="./logs", show_default=True)
@click.option("--logging_steps", default=50, show_default=True)
@click.option("--save_steps", default=500, show_default=True)
@click.option("--evaluation_strategy", default="steps", show_default=True)
def main(
        train_dataset_folder,
        train_initial_data,
        test_dataset_folder,
        test_initial_data,
        k_folds,
        train_epochs,
        batch_size,
        warmup_steps,
        learning_rate,
        weight_decay,
        logging_dir,
        logging_steps,
        save_steps,
        evaluation_strategy,
        ) -> None:
    train_dataset = ComplexityDataset(
        prepare_dataset(train_dataset_folder, train_initial_data),
        get_tokenizer("DeepPavlov/rubert-base-cased"))
    if test_dataset_folder != train_dataset_folder:
        test_dataset = ComplexityDataset(
        prepare_dataset(test_dataset_folder, test_initial_data),
        get_tokenizer("DeepPavlov/rubert-base-cased"))
    else:
        test_dataset = None


    model = get_model("DeepPavlov/rubert-base-cased")


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
    run_experiment(model, train_dataset, test_dataset,
                   k_folds=k_folds, training_args=training_args)

if __name__=="__main__":
    main()
