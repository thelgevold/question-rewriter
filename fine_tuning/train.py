import logging
import json
from pathlib import Path
import re
from typing import Any, Iterable

from data.data_loader import load_training_payload, split_train_eval_payload
logging.getLogger(
    "vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe"
).disabled = True
import unsloth
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from fine_tuning.config import CONFIG
from fine_tuning.handlers.ollama_handler import export_gguf_for_ollama, save_adapter_and_merged
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


def build_user_prompt(record: dict[str, Any]) -> str:
    conversation_lines = list(iter_conversation_lines(record))
    return (
        f"{USER_TASK_INTRO}\n\n"
        f"Conversation:\n" + "\n".join(conversation_lines) + "\n\n"
        "Standalone question:"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_accuracy_summary(name: str, report: dict[str, Any]) -> None:
    summary = (
        f"{name} normalized exact match: "
        f"{report['normalized_exact_match_count']}/{report['count']} "
        f"({report['normalized_exact_match_accuracy'] * 100:.2f}%)"
    )
    LOGGER.info(summary)
    print(summary, flush=True)

    semantic_summary = (
        f"{name} semantic pass @ {CONFIG.semantic_similarity_threshold:.2f}: "
        f"{report['semantic_match_count']}/{report['count']} "
        f"({report['semantic_match_accuracy'] * 100:.2f}%), "
        f"avg cosine={report['average_semantic_similarity']:.4f}"
    )
    LOGGER.info(semantic_summary)
    print(semantic_summary, flush=True)


def apply_chat_template_without_thinking(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }

    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )


def build_trainer_args(output_dir: Path, bf16_supported: bool) -> SFTConfig:
    return SFTConfig(
        per_device_train_batch_size=CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
        warmup_steps=CONFIG.warmup_steps,
        num_train_epochs=CONFIG.num_train_epochs,
        learning_rate=CONFIG.learning_rate,
        logging_steps=CONFIG.logging_steps,
        save_steps=CONFIG.save_steps,
        save_total_limit=2,
        seed=CONFIG.seed,
        output_dir=str(output_dir),
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        report_to="none",
        dataset_text_field="text",
        max_length=CONFIG.max_seq_length,
        packing=False,
        eval_strategy="steps",
        eval_steps=CONFIG.eval_steps,
    )


def build_training_examples(records: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    formatted_rows: list[dict[str, str]] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Example {index} must be a JSON object.")

        target_question = extract_target_question(record)
        user_prompt = build_user_prompt(record)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_question},
        ]

        text = apply_chat_template_without_thinking(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )

        formatted_rows.append({"text": text})

    return Dataset.from_list(formatted_rows)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def strip_think_fragments(value: str) -> str:
    without_think_blocks = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    return without_think_blocks.strip()


def generate_rewrite(model: Any, tokenizer: Any, record: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]
    prompt = apply_chat_template_without_thinking(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    generated_tokens = generated[0][prompt_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    prediction = strip_think_fragments(prediction)
    return prediction


def build_prediction_report(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    semantic_model: SentenceTransformer,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact_matches = 0
    semantic_matches = 0
    semantic_similarity_total = 0.0

    for index, record in enumerate(records, start=1):
        expected = extract_target_question(record)
        predicted = generate_rewrite(model, tokenizer, record)
        is_exact_match = normalize_text(predicted) == normalize_text(expected)
        if is_exact_match:
            exact_matches += 1

        embeddings = semantic_model.encode(
            [expected, predicted],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        semantic_similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        semantic_similarity_total += semantic_similarity
        is_semantic_match = semantic_similarity >= CONFIG.semantic_similarity_threshold
        if is_semantic_match:
            semantic_matches += 1

        rows.append(
            {
                "index": index,
                "test_group": record.get("test_group"),
                "pattern_note": record.get("pattern_note"),
                "messages": record["messages"],
                "expected_rewrite": expected,
                "predicted_rewrite": predicted,
                "normalized_exact_match": is_exact_match,
                "semantic_similarity": semantic_similarity,
                "semantic_match": is_semantic_match,
            }
        )

    count = len(rows)
    return {
        "count": count,
        "normalized_exact_match_count": exact_matches,
        "normalized_exact_match_accuracy": (exact_matches / count) if count else 0.0,
        "semantic_match_count": semantic_matches,
        "semantic_match_accuracy": (semantic_matches / count) if count else 0.0,
        "average_semantic_similarity": (semantic_similarity_total / count) if count else 0.0,
        "predictions": rows,
    }


def load_merged_export_model(merged_dir: Path) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=CONFIG.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main() -> None:
    logging.basicConfig(
        level=CONFIG.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    training_data_path = CONFIG.training_data_path
    test_data_path = CONFIG.test_data_path
    output_dir = CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading training records from %s", training_data_path)
    training_records = load_training_payload(training_data_path)
    LOGGER.info("Loaded %s training-source examples", len(training_records))
    LOGGER.info("Loading static test records from %s", test_data_path)
    test_records = load_training_payload(test_data_path)
    LOGGER.info("Loaded %s static test examples", len(test_records))
    LOGGER.info("Loading semantic similarity model %s", CONFIG.semantic_similarity_model_name)
    semantic_model = SentenceTransformer(CONFIG.semantic_similarity_model_name)

    split_records = split_train_eval_payload(
        training_records,
        eval_size=CONFIG.eval_split_ratio,
        seed=CONFIG.seed,
    )
    LOGGER.info(
        "Split training-source records into train=%s eval=%s",
        len(split_records["train"]),
        len(split_records["eval"]),
    )

    splits_dir = output_dir / "splits"
    write_json(splits_dir / "train.json", split_records["train"])
    write_json(splits_dir / "eval.json", split_records["eval"])
    write_json(splits_dir / "test.json", test_records)

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

    train_dataset = build_training_examples(split_records["train"], tokenizer)
    eval_dataset = build_training_examples(split_records["eval"], tokenizer)
    test_dataset = build_training_examples(test_records, tokenizer)
    LOGGER.info(
        "Prepared formatted datasets train=%s eval=%s test=%s",
        len(train_dataset),
        len(eval_dataset),
        len(test_dataset),
    )
    bf16_supported = is_bfloat16_supported()
    trainer_args = build_trainer_args(output_dir / "checkpoints", bf16_supported)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=trainer_args,
    )

    LOGGER.info("Starting training")
    trainer.train()
    LOGGER.info("Training complete")

    eval_metrics = trainer.evaluate()
    test_trainer = SFTTrainer(
        model=trainer.model,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=build_trainer_args(output_dir / "test-eval", bf16_supported),
    )
    test_metrics = test_trainer.evaluate(metric_key_prefix="test")
    reports_dir = output_dir / "reports"
    write_json(reports_dir / "eval_metrics.json", eval_metrics)
    write_json(reports_dir / "test_metrics.json", test_metrics)
    LOGGER.info("Saved eval metrics to %s", reports_dir / "eval_metrics.json")
    LOGGER.info("Saved test metrics to %s", reports_dir / "test_metrics.json")

    FastLanguageModel.for_inference(trainer.model)
    eval_predictions = build_prediction_report(trainer.model, tokenizer, split_records["eval"], semantic_model)
    test_predictions = build_prediction_report(trainer.model, tokenizer, test_records, semantic_model)
    print_accuracy_summary("Eval", eval_predictions)
    print_accuracy_summary("Test", test_predictions)
    write_json(reports_dir / "eval_predictions.json", eval_predictions)
    write_json(reports_dir / "test_predictions.json", test_predictions)
    LOGGER.info("Saved eval predictions to %s", reports_dir / "eval_predictions.json")
    LOGGER.info("Saved test predictions to %s", reports_dir / "test_predictions.json")

    trainer.save_state()

    export_paths = save_adapter_and_merged(
        model=trainer.model,
        tokenizer=tokenizer,
        export_root=output_dir,
        export_base_model_name=CONFIG.export_base_model_name,
    )

    merged_model, merged_tokenizer = load_merged_export_model(export_paths["merged_dir"])

    export_gguf_for_ollama(
        model=merged_model,
        tokenizer=merged_tokenizer,
        export_root=output_dir,
        quantization=CONFIG.ollama_gguf_quantization,
    )

    del test_trainer
    del model
    del merged_model
    del trainer
    del semantic_model
    torch.cuda.empty_cache()

    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
