from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    training_data_path: Path = Path("/workspace/data/training-questions.json")
    output_model_prefix: str = "question-rewriter"
    base_model_slug: str = "qwen3-0.6b"
    base_model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit"
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    num_train_epochs: float = 5
    warmup_steps: int = 10
    logging_steps: int = 1
    save_steps: int = 50
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 3407
    ollama_gguf_quantization: str = "Q4_K_M"
    log_level: str = "INFO"

    @property
    def output_model_name(self) -> str:
        return f"{self.output_model_prefix}-{self.base_model_slug}"

    @property
    def output_dir(self) -> Path:
        return Path("/workspace/outputs") / self.output_model_name


CONFIG = TrainingConfig()
