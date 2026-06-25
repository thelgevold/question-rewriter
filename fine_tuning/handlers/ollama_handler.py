import logging
from pathlib import Path
from typing import Any
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("question_rewriter_fine_tune")
GGUF_FILENAME_PREFIX = "question-rewriter-"


def _find_exported_gguf(export_root: Path, preferred_dir: Path) -> Path | None:
    direct_candidates = sorted(preferred_dir.glob("*.gguf"))
    if direct_candidates:
        return direct_candidates[0]

    recursive_candidates = sorted(
        candidate for candidate in export_root.rglob("*.gguf") if candidate.is_file()
    )
    if recursive_candidates:
        return max(recursive_candidates, key=lambda path: path.stat().st_mtime)

    return None


def _find_generated_modelfile(export_root: Path) -> Path | None:
    direct_candidate = export_root / "gguf_gguf" / "Modelfile"
    if direct_candidate.is_file():
        return direct_candidate

    recursive_candidates = sorted(
        candidate
        for candidate in export_root.rglob("Modelfile")
        if candidate.is_file() and candidate.parent != export_root
    )
    if recursive_candidates:
        return max(recursive_candidates, key=lambda path: path.stat().st_mtime)

    return None


def _build_ollama_modelfile(
    gguf_path: Path,
    export_root: Path,
    generated_modelfile_path: Path | None,
) -> None:
    modelfile_path = export_root / "Modelfile"
    if generated_modelfile_path and generated_modelfile_path.is_file():
        modelfile_content = generated_modelfile_path.read_text(encoding="utf-8")
        modelfile_lines = modelfile_content.splitlines()
        rewritten_lines = [
            f"FROM ./{gguf_path.name}" if line.strip().startswith("FROM ") else line
            for line in modelfile_lines
        ]
        rewritten_content = "\n".join(rewritten_lines) + "\n"
        modelfile_path.write_text(rewritten_content, encoding="utf-8")
        return

    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM ./{gguf_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _get_canonical_gguf_path(export_root: Path, gguf_file: Path) -> Path:
    filename = gguf_file.name
    if not filename.lower().startswith(GGUF_FILENAME_PREFIX):
        filename = f"{GGUF_FILENAME_PREFIX}{filename}"
    return export_root / filename


def _merge_adapter_into_full_precision_base(
    adapter_dir: Path,
    merged_dir: Path,
    export_base_model_name: str,
) -> None:
    LOGGER.info(
        "Loading full-precision export base model %s for adapter merge",
        export_base_model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        export_base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    generation_config = getattr(merged_model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(str(merged_dir))


def save_adapter_and_merged(
    model: Any,
    tokenizer: Any,
    export_root: Path,
    export_base_model_name: str,
) -> dict[str, Path]:
    merged_dir = export_root / "merged"
    adapter_dir = export_root / "adapter"

    merged_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving adapter weights to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    LOGGER.info("Saving merged 16-bit model to %s via standard PEFT merge", merged_dir)
    _merge_adapter_into_full_precision_base(
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        export_base_model_name=export_base_model_name,
    )

    return {
        "adapter_dir": adapter_dir,
        "merged_dir": merged_dir,
    }


def export_gguf_for_ollama(model: Any, tokenizer: Any, export_root: Path, quantization: str) -> None:
    gguf_dir = export_root / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Exporting GGUF model to %s with quantization %s", gguf_dir, quantization)
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=quantization,
    )

    gguf_file = _find_exported_gguf(export_root, gguf_dir)
    if gguf_file is None:
        LOGGER.warning("GGUF export was not produced, so no Ollama Modelfile was created.")
        return

    canonical_gguf_path = _get_canonical_gguf_path(export_root, gguf_file)
    if gguf_file.resolve() != canonical_gguf_path.resolve():
        LOGGER.info("Copying GGUF artifact to %s for Ollama compatibility", canonical_gguf_path)
        shutil.copy2(gguf_file, canonical_gguf_path)
    else:
        LOGGER.info("Using GGUF artifact at %s", canonical_gguf_path)

    generated_modelfile_path = _find_generated_modelfile(export_root)
    _build_ollama_modelfile(canonical_gguf_path, export_root, generated_modelfile_path)
    LOGGER.info(
        "Prepared Ollama GGUF artifact at %s using %s quantization",
        canonical_gguf_path,
        quantization,
    )
    LOGGER.info(
        "Prepared Ollama Modelfile at %s pointing to %s",
        export_root / "Modelfile",
        canonical_gguf_path.name,
    )
