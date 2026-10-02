#!/usr/bin/env python3
"""Pack a Transformers checkpoint into a new Ollama tag.

Examples:
  # Default: F16 GGUF via llama.cpp, then ollama create
  venv/bin/python scripts/hf_to_ollama.py \\
    --checkpoint models/bridge-qwen05-edited \\
    --tag aiasylum-qwen05-edited

  # Quantize after conversion
  venv/bin/python scripts/hf_to_ollama.py \\
    --checkpoint models/bridge-qwen05-edited \\
    --tag aiasylum-qwen05-edited-q8 --quant q8_0

  # Skip GGUF: Ollama experimental safetensors import (0.32+)
  venv/bin/python scripts/hf_to_ollama.py \\
    --checkpoint models/bridge-qwen05-edited \\
    --tag aiasylum-qwen05-edited --experimental-ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivasecuris.aiasylum.bridge.convert import (
    BridgeError,
    build_modelfile,
    chat_template_from_tokenizer_config,
    hf_to_gguf,
    ollama_create,
    ollama_create_from_safetensors,
    quantize_gguf,
    validate_hf_checkpoint,
    write_modelfile,
)
from vivasecuris.aiasylum.bridge.provenance import read_bridge_manifest, write_bridge_manifest


def _read_sidecar(checkpoint: Path, name: str) -> str | None:
    path = checkpoint / name
    if path.is_file():
        text = path.read_text(errors="replace").strip()
        return text or None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Transformers checkpoint directory")
    parser.add_argument("--tag", required=True, help="New Ollama model tag to create")
    parser.add_argument(
        "--quant",
        default="f16",
        help="GGUF output type / post-quantize target: f16 (default), q8_0, q4_k_m, …",
    )
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp with convert_hf_to_gguf.py")
    parser.add_argument("--work-dir", default=None, help="Work directory (default runs/ollama-bridge/<tag>)")
    parser.add_argument(
        "--experimental-ollama",
        action="store_true",
        help="Use ollama create --experimental FROM safetensors (skip llama.cpp GGUF)",
    )
    parser.add_argument("--system", default=None, help="Override SYSTEM prompt in Modelfile")
    parser.add_argument("--no-template", action="store_true", help="Do not embed a TEMPLATE in the Modelfile")
    parser.add_argument("--skip-create", action="store_true", help="Write GGUF/Modelfile only; do not run ollama create")
    args = parser.parse_args(argv)

    try:
        checkpoint = validate_hf_checkpoint(args.checkpoint)
        prior = read_bridge_manifest(checkpoint) or {}
        system = args.system or _read_sidecar(checkpoint, "ollama-system.txt") or prior.get("ollama_system")
        template = None
        if not args.no_template:
            template = _read_sidecar(checkpoint, "ollama-template.txt")
            if template is None:
                template = chat_template_from_tokenizer_config(checkpoint)

        work = Path(args.work_dir) if args.work_dir else ROOT / "runs" / "ollama-bridge" / args.tag.replace(":", "_")
        work.mkdir(parents=True, exist_ok=True)

        if args.experimental_ollama:
            if args.skip_create:
                raise BridgeError("--skip-create cannot be combined with --experimental-ollama")
            print(f"Creating {args.tag} from safetensors via ollama --experimental", flush=True)
            modelfile = ollama_create_from_safetensors(
                args.tag,
                checkpoint,
                template=template,
                system=system,
                work_dir=work,
            )
            gguf_path = None
            quant_used = "experimental_safetensors"
        else:
            quant = args.quant.lower().strip()
            # convert_hf_to_gguf outtypes vs llama-quantize types
            convert_outtype = quant if quant in {"f32", "f16", "bf16", "q8_0", "auto"} else "f16"
            gguf_path = work / f"{checkpoint.name}-{convert_outtype}.gguf"
            print(f"Converting {checkpoint} → {gguf_path} (outtype={convert_outtype})", flush=True)
            hf_to_gguf(checkpoint, gguf_path, outtype=convert_outtype, llama_cpp_dir=args.llama_cpp_dir)
            quant_used = convert_outtype
            if quant not in {"f32", "f16", "bf16", "q8_0", "auto"} and quant != convert_outtype:
                quantized = work / f"{checkpoint.name}-{quant}.gguf"
                print(f"Quantizing → {quantized} ({quant})", flush=True)
                gguf_path = quantize_gguf(gguf_path, quantized, quant)
                quant_used = quant

            modelfile_text = build_modelfile(gguf_path, template=template, system=system)
            modelfile = write_modelfile(work / "Modelfile", modelfile_text)
            # Place GGUF next to Modelfile with the name FROM expects (already same dir).
            if gguf_path.parent.resolve() != work.resolve():
                raise BridgeError("internal: GGUF must live beside Modelfile")
            if not args.skip_create:
                print(f"ollama create {args.tag}", flush=True)
                ollama_create(args.tag, modelfile)

        write_bridge_manifest(
            work,
            direction="hf_to_ollama",
            payload={
                "checkpoint": str(checkpoint),
                "tag": args.tag,
                "quant": quant_used,
                "gguf_path": str(gguf_path) if gguf_path else None,
                "modelfile": str(modelfile),
                "experimental_ollama": bool(args.experimental_ollama),
                "system_set": bool(system),
                "template_set": bool(template),
                "source_bridge": prior,
            },
        )
        print(f"Ready: ollama run {args.tag}", flush=True)
        return 0
    except BridgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
