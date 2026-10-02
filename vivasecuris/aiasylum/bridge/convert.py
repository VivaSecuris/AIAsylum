"""Checkpoint validation, GGUF ↔ HF conversion, and Modelfile rendering."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from vivasecuris.aiasylum.bridge.ollama_store import BridgeError

# Pinned llama.cpp revision used when cloning for convert_hf_to_gguf.py.
# Update deliberately and record the change in docs/OLLAMA_TRANSFORMERS_BRIDGE.md.
PINNED_LLAMA_CPP_REV = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"

DEFAULT_LLAMA_CPP_HINTS = (
    "runs/ollama-bridge/llama.cpp",
    "third_party/llama.cpp",
    os.path.expanduser("~/llama.cpp"),
)


def validate_hf_checkpoint(root: str | Path) -> Path:
    """Require config, tokenizer, and non-empty weight shards."""
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise BridgeError(f"Checkpoint must be a real directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise BridgeError(f"Missing config.json in {path}")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"Invalid config.json in {path}: {exc}") from exc
    if not config.get("model_type"):
        raise BridgeError(f"Missing model_type in {path}/config.json")
    if not (path / "tokenizer_config.json").is_file():
        raise BridgeError(f"Missing tokenizer_config.json in {path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "spiece.model")):
        raise BridgeError(f"No complete tokenizer found in {path}")
    indexes = [
        path / name
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (path / name).is_file()
    ]
    if indexes:
        for index in indexes:
            weights = json.loads(index.read_text()).get("weight_map") or {}
            if not weights:
                raise BridgeError(f"Empty weight map: {index}")
            for name in set(weights.values()):
                shard = path / name
                if not shard.is_file() or shard.stat().st_size == 0:
                    raise BridgeError(f"Missing or empty weight shard {shard}")
    elif not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise BridgeError(f"No complete weights found in {path}")
    return path.resolve()


def read_gguf_file_type(gguf_path: str | Path) -> Optional[int]:
    """Return GGUF ``general.file_type`` when readable."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise BridgeError(
            "The gguf package is required to inspect Ollama blobs. "
            "Install it in the project venv: pip install gguf"
        ) from exc
    reader = GGUFReader(str(gguf_path))
    field = reader.fields.get("general.file_type")
    if field is None or not field.data:
        return None
    try:
        return int(field.parts[field.data[0]][0])
    except Exception:
        return None


def gguf_quant_label(file_type: Optional[int]) -> str:
    # Matches ggml file type enum used by GGUF / llama.cpp.
    labels = {
        0: "f32",
        1: "f16",
        2: "q4_0",
        3: "q4_1",
        7: "q8_0",
        8: "q5_0",
        9: "q5_1",
        10: "q2_k",
        11: "q3_k",
        12: "q4_k",
        13: "q5_k",
        14: "q6_k",
        15: "q4_k_m",
        16: "bf16",
    }
    if file_type is None:
        return "unknown"
    return labels.get(int(file_type), f"file_type_{file_type}")


def gguf_to_hf(gguf_path: str | Path, out_dir: str | Path, dtype: str = "float16") -> Path:
    """Dequantize a GGUF file into a Transformers safetensors directory.

    This path is lossy when the GGUF is quantized. Prefer ``--hf-id`` for surgery.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gguf = Path(gguf_path).resolve()
    if not gguf.is_file():
        raise BridgeError(f"GGUF not found: {gguf}")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise BridgeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    work = out.parent / f".gguf-work-{out.name}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    linked = work / gguf.name
    if linked.exists():
        linked.unlink()
    try:
        os.link(gguf, linked)
    except OSError:
        shutil.copy2(gguf, linked)

    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype.lower())
    if torch_dtype is None:
        raise BridgeError(f"Unsupported dtype for GGUF import: {dtype}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(work),
            gguf_file=linked.name,
            dtype=torch_dtype,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(work),
            gguf_file=linked.name,
            trust_remote_code=False,
        )
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
    except Exception as exc:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        raise BridgeError(
            f"Failed to convert GGUF to Hugging Face format: {exc}. "
            f"If this architecture is unsupported by Transformers' GGUF loader, "
            f"pass --hf-id with the publisher checkpoint instead."
        ) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return validate_hf_checkpoint(out)


def find_llama_cpp_dir(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not (path / "convert_hf_to_gguf.py").is_file():
            raise BridgeError(f"convert_hf_to_gguf.py not found under {path}")
        return path
    env = os.environ.get("LLAMA_CPP_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    # Prefer repo-relative runs/ cache when invoked from the project root.
    here = Path.cwd()
    for hint in DEFAULT_LLAMA_CPP_HINTS:
        candidates.append((here / hint).resolve() if not os.path.isabs(hint) else Path(hint))
        candidates.append(Path(__file__).resolve().parents[3] / hint)
    for path in candidates:
        if (path / "convert_hf_to_gguf.py").is_file():
            return path.resolve()
    raise BridgeError(
        "llama.cpp convert_hf_to_gguf.py not found. Clone a pinned tree, for example:\n"
        f"  git clone https://github.com/ggml-org/llama.cpp runs/ollama-bridge/llama.cpp\n"
        f"  cd runs/ollama-bridge/llama.cpp && git checkout {PINNED_LLAMA_CPP_REV}\n"
        "Or set LLAMA_CPP_DIR / pass --llama-cpp-dir. "
        "Alternatively use --experimental-ollama to skip GGUF conversion on Ollama ≥0.32."
    )


def hf_to_gguf(
    checkpoint: str | Path,
    outfile: str | Path,
    *,
    outtype: str = "f16",
    llama_cpp_dir: Optional[str | Path] = None,
    python_exe: Optional[str] = None,
) -> Path:
    """Convert a Hugging Face checkpoint directory to GGUF via llama.cpp."""
    src = validate_hf_checkpoint(checkpoint)
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    llama_dir = find_llama_cpp_dir(llama_cpp_dir)
    script = llama_dir / "convert_hf_to_gguf.py"
    py = python_exe or sys.executable
    cmd = [
        py,
        str(script),
        str(src),
        "--outfile",
        str(out),
        "--outtype",
        outtype,
    ]
    env = os.environ.copy()
    # Prefer the convert tree's vendored gguf-py when present.
    gguf_py = llama_dir / "gguf-py"
    if gguf_py.is_dir():
        env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(llama_dir))
    except FileNotFoundError as exc:
        raise BridgeError(f"Cannot run converter with {py}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        hint = ""
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            hint = " Install sentencepiece in the venv (`pip install sentencepiece`)."
        raise BridgeError(
            f"convert_hf_to_gguf.py failed with exit {exc.returncode}.{hint} "
            f"Command: {' '.join(cmd)}"
        ) from exc
    if not out.is_file() or out.stat().st_size == 0:
        raise BridgeError(f"Converter did not write a GGUF file at {out}")
    return out.resolve()


def find_llama_quantize(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise BridgeError(f"llama-quantize not found: {path}")
        return path.resolve()
    env = os.environ.get("LLAMA_QUANTIZE")
    if env and Path(env).is_file():
        return Path(env).resolve()
    which = shutil.which("llama-quantize")
    if which:
        return Path(which).resolve()
    raise BridgeError(
        "llama-quantize not found on PATH. Install llama.cpp "
        "(for example `brew install llama.cpp`) or set LLAMA_QUANTIZE."
    )


def quantize_gguf(
    src_gguf: str | Path,
    dest_gguf: str | Path,
    quant: str,
    *,
    llama_quantize: Optional[str] = None,
) -> Path:
    """Quantize an F16/F32 GGUF to ``quant`` (for example ``q8_0``, ``q4_k_m``)."""
    src = Path(src_gguf)
    if not src.is_file():
        raise BridgeError(f"Source GGUF not found: {src}")
    dest = Path(dest_gguf)
    dest.parent.mkdir(parents=True, exist_ok=True)
    binary = find_llama_quantize(llama_quantize)
    # llama-quantize: input output type
    cmd = [str(binary), str(src), str(dest), quant.upper()]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # Some builds want lowercase / different enum names — retry as given.
        alt = [str(binary), str(src), str(dest), quant]
        try:
            subprocess.run(alt, check=True)
        except subprocess.CalledProcessError as exc2:
            raise BridgeError(
                f"llama-quantize failed ({exc.returncode}/{exc2.returncode}) for {quant}"
            ) from exc2
    if not dest.is_file() or dest.stat().st_size == 0:
        raise BridgeError(f"Quantize did not write {dest}")
    return dest.resolve()


def build_modelfile(
    gguf_path: str | Path,
    *,
    template: Optional[str] = None,
    system: Optional[str] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> str:
    """Render an Ollama Modelfile that loads a local GGUF."""
    gguf = Path(gguf_path)
    # Ollama resolves FROM relative to the Modelfile directory.
    lines = [f"FROM {gguf.name}"]
    if system:
        # PARAMETER-less SYSTEM block; escape triple quotes if needed.
        cleaned = system.strip()
        if '"""' in cleaned:
            cleaned = cleaned.replace('"""', "'''")
        lines.append(f'SYSTEM """{cleaned}"""')
    if template:
        cleaned = template.strip()
        if '"""' in cleaned:
            # Fall back to single-line PARAMETER-style avoidance: write as TEMPLATE with quotes
            cleaned = cleaned.replace('"""', "''")
        lines.append(f'TEMPLATE """{cleaned}"""')
    for key, value in (parameters or {}).items():
        lines.append(f"PARAMETER {key} {value}")
    return "\n".join(lines) + "\n"


def write_modelfile(path: str | Path, content: str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest.resolve()


def ollama_create(
    tag: str,
    modelfile: str | Path,
    *,
    quantize: Optional[str] = None,
    experimental: bool = False,
    ollama_bin: str = "ollama",
) -> None:
    """Run ``ollama create`` for ``tag`` from ``modelfile``."""
    tag = tag.strip()
    if not tag:
        raise BridgeError("Ollama create tag is empty")
    if tag.endswith(":cloud") or tag.split(":")[-1] == "cloud":
        raise BridgeError("Refusing to create a :cloud tag locally")
    mf = Path(modelfile)
    if not mf.is_file():
        raise BridgeError(f"Modelfile not found: {mf}")
    cmd: list[str] = [ollama_bin, "create", tag, "-f", str(mf)]
    if quantize:
        cmd.extend(["--quantize", quantize])
    if experimental:
        cmd.append("--experimental")
    try:
        subprocess.run(cmd, check=True, cwd=str(mf.parent))
    except FileNotFoundError as exc:
        raise BridgeError("ollama CLI not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise BridgeError(f"ollama create failed with exit {exc.returncode}: {' '.join(cmd)}") from exc


def ollama_create_from_safetensors(
    tag: str,
    checkpoint: str | Path,
    *,
    template: Optional[str] = None,
    system: Optional[str] = None,
    work_dir: str | Path,
    ollama_bin: str = "ollama",
) -> Path:
    """Create an Ollama model from a safetensors directory using ``--experimental``."""
    src = validate_hf_checkpoint(checkpoint)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Modelfile FROM must point at the checkpoint directory.
    modelfile = work / "Modelfile"
    lines = [f"FROM {src.resolve()}"]
    if system:
        cleaned = system.strip().replace('"""', "'''")
        lines.append(f'SYSTEM """{cleaned}"""')
    if template:
        cleaned = template.strip().replace('"""', "''")
        lines.append(f'TEMPLATE """{cleaned}"""')
    modelfile.write_text("\n".join(lines) + "\n")
    ollama_create(tag, modelfile, experimental=True, ollama_bin=ollama_bin)
    return modelfile.resolve()


def chat_template_from_tokenizer_config(checkpoint: str | Path) -> Optional[str]:
    path = Path(checkpoint) / "tokenizer_config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    template = data.get("chat_template")
    return template if isinstance(template, str) and template.strip() else None


def snapshot_download_to(repo_id: str, out_dir: str | Path, revision: Optional[str] = None) -> Path:
    """Download a Hugging Face snapshot into ``out_dir``."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise BridgeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BridgeError("huggingface_hub is required for --hf-id downloads") from exc
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out),
    )
    return validate_hf_checkpoint(out)
