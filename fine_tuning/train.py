import logging
from typing import Any, Iterable

from data.data_loader import load_training_payload
logging.getLogger(
    "vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe"
).disabled = True
import unsloth
from datasets import Dataset
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from fine_tuning.config import CONFIG
from fine_tuning.handlers.ollama_handler import export_for_ollama
from fine_tuning.prompts import SYSTEM_PROMPT, USER_TASK_INTRO


LOGGER = logging.getLogger("question_rewriter_fine_tune")


def is_bfloat16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def iter_conversation_lines(record: dict[str, Any]) -> Iterable[str]:
    messages = record.get("messages")
        
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f'Example message {index} must be an object with "role" and "content".')

        role = message.get("role")
    
        content = message.get("content")

        yield f"{role.title()}: {content.strip()}"


def extract_target_question(record: dict[str, Any]) -> str:
    rewrite = record.get("rewrite")
  
    return rewrite.strip()


def build_training_examples(records: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    formatted_rows: list[dict[str, str]] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Example {index} must be a JSON object.")

        conversation_lines = list(iter_conversation_lines(record))

        target_question = extract_target_question(record)

        user_prompt = (
            f"{USER_TASK_INTRO}\n\n"
            f"Conversation:\n" + "\n".join(conversation_lines) + "\n\n"
            "Standalone question:"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_question},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        formatted_rows.append({"text": text})

    return Dataset.from_list(formatted_rows)


def main() -> None:
    logging.basicConfig(
        level=CONFIG.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    training_data_path = CONFIG.training_data_path
    output_dir = CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading training records from %s", training_data_path)
    records = load_training_payload(training_data_path)
    LOGGER.info("Loaded %s training examples", len(records))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG.base_model_name,
        max_seq_length=CONFIG.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=CONFIG.lora_alpha,
        lora_dropout=CONFIG.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG.seed,
    )

    dataset = build_training_examples(records, tokenizer)
    LOGGER.info("Prepared %s formatted training rows", len(dataset))
    bf16_supported = is_bfloat16_supported()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=CONFIG.per_device_train_batch_size,
            gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
            warmup_steps=CONFIG.warmup_steps,
            num_train_epochs=CONFIG.num_train_epochs,
            learning_rate=CONFIG.learning_rate,
            logging_steps=CONFIG.logging_steps,
            save_steps=CONFIG.save_steps,
            save_total_limit=2,
            seed=CONFIG.seed,
            output_dir=str(output_dir / "checkpoints"),
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            fp16=not bf16_supported,
            bf16=bf16_supported,
            report_to="none",
            dataset_text_field="text",
            max_length=CONFIG.max_seq_length,
            packing=False,
        ),
    )

    LOGGER.info("Starting training")
    trainer.train()
    LOGGER.info("Training complete")

    adapter_output_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))
    trainer.save_state()

    export_for_ollama(
        model=trainer.model,
        tokenizer=tokenizer,
        export_root=output_dir,
        quantization=CONFIG.ollama_gguf_quantization,
    )

    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
