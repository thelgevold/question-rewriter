import json
from pathlib import Path
from typing import Any


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

    records = payload

    if not records:
        raise ValueError("Training data contains no examples.")

    return records
