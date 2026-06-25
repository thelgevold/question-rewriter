import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


def load_training_payload(training_data_path: Path) -> list[dict[str, Any]]:
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {training_data_path}")

    if training_data_path.stat().st_size == 0:
        raise ValueError(
            "Training data file is empty. Populate it with a JSON array of examples "
            'that each contain "messages" and "rewrite".'
        )

    with training_data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            'Unsupported JSON structure. Expected a JSON array where each item has '
            '"messages" and "rewrite".'
        )

    if not payload:
        raise ValueError("Training data contains no examples.")

    return payload


def split_train_eval_payload(
    records: list[dict[str, Any]],
    *,
    eval_size: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < eval_size < 1:
        raise ValueError(f"eval_size must be between 0 and 1, but received {eval_size}.")

    if len(records) < 2:
        raise ValueError("Training data must contain at least 2 examples to create train/eval splits.")

    train_records, eval_records = train_test_split(
        records,
        test_size=eval_size,
        random_state=seed,
        shuffle=True,
    )

    if not train_records or not eval_records:
        raise ValueError(
            "Train/eval split produced an empty dataset. Add more examples or adjust the eval split ratio."
        )

    return {
        "train": train_records,
        "eval": eval_records,
    }
