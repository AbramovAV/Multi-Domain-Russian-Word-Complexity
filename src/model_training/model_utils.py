"""
Module with utils for model loading
"""
from typing import Dict

from transformers import (AutoModelForSequenceClassification, PreTrainedModel,
                          Trainer)

from src.model_training.dataset_utils import compute_metrics


def model_init(
        trial: int  # pylint: disable=unused-argument
        ) -> PreTrainedModel:
    """
    Initializes pretrained model for the new Optuna optimization trial.

    Args:
        trial: index of the new optimization trial

    Returns:
        freshly initialized pretrained model
    """
    return AutoModelForSequenceClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased",
        from_tf=False,
        num_labels=1
    )


def optuna_hp_space(trial: int) -> Dict:  # pylint: disable=unused-argument
    """
    Defines search space for hyperparameters optimization.

    Args:
        trial: index of the new optimization trial

    Returns:
        dictionary with hyperparameters and allowed search ranges.
    """
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-6, 1e-2, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]),
        "warmup_steps": trial.suggest_categorical(
            "warmup_steps", [1, 5, 10, 15, 20]),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-1, log=True)
    }


def compute_objective(metrics: Dict[str, float]) -> float:
    """
    Computes optimization objective for hyperparameters optimization.

    Args:
        metrics: dictionary with evaluation metrics

    Returns:
        sum of evaluation metrics as optimization objective
    """
    corr = 0
    mae = 0
    if "eval_correlation_coefficient" in metrics:
        corr = -metrics["eval_correlation_coefficient"]
    if "eval_mean_absolute_error" in metrics:
        mae = metrics["eval_mean_absolute_error"]
    return corr + mae


def run_hyperparameter_optimization(
        train_dataset, eval_dataset, training_args):
    """
    Runs hyperparameter optimization for 50 trials.
    """
    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=50,
        compute_objective=compute_objective

    )
    hyperparameters = best_trial.hyperparameters
    for param in hyperparameters:
        setattr(training_args, param, hyperparameters[param])
    return training_args
