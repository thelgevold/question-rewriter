import logging
from pathlib import Path
from typing import Any
import shutil

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


def _build_ollama_modelfile(
    gguf_path: Path,
    export_root: Path,
    temperature: float = 0,
) -> None:
    modelfile_path = export_root / "Modelfile"
    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM ./{gguf_path.name}",
                f"PARAMETER temperature {temperature}",
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


def export_for_ollama(model: Any, tokenizer: Any, export_root: Path, quantization: str) -> None:
    merged_dir = export_root / "merged"
    gguf_dir = export_root / "gguf"
    adapter_dir = export_root / "adapter"

    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving adapter weights to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    LOGGER.info("Saving merged 16-bit model to %s", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

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

    _build_ollama_modelfile(canonical_gguf_path, export_root)
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
