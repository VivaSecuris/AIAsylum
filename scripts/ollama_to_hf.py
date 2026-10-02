#!/usr/bin/env python3
"""Export a local Ollama model to a Transformers checkpoint for weight surgery.

Examples:
  # Preferred surgery base: known HF twin (still records the Ollama blob digest)
  venv/bin/python scripts/ollama_to_hf.py \\
    --model qwen2.5:0.5b --out models/bridge-qwen05 \\
    --hf-id Qwen/Qwen2.5-0.5B-Instruct

  # Lossy path: dequantize the Ollama GGUF blob into safetensors
  venv/bin/python scripts/ollama_to_hf.py \\
    --model qwen2.5:0.5b --out models/bridge-qwen05-from-gguf
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
    gguf_quant_label,
    gguf_to_hf,
    read_gguf_file_type,
    snapshot_download_to,
)
from vivasecuris.aiasylum.bridge.ollama_store import (
    assert_local_ollama_model,
    export_weight_gguf,
    ollama_models_root,
)
from vivasecuris.aiasylum.bridge.provenance import write_bridge_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local Ollama model name (e.g. qwen2.5:0.5b)")
    parser.add_argument("--out", required=True, help="Output directory under models/ (or any path)")
    parser.add_argument("--hf-id", default=None, help="Download this HF repo instead of dequantizing GGUF")
    parser.add_argument("--revision", default=None, help="Optional HF revision when using --hf-id")
    parser.add_argument("--ollama-models", default=None, help="Override Ollama models root")
    parser.add_argument("--work-dir", default=None, help="Directory for exported GGUF (default: runs/ollama-bridge/)")
    parser.add_argument("--dtype", default="float16", help="Torch dtype for GGUF dequant import (default float16)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    try:
        models_root = ollama_models_root(args.ollama_models) if args.ollama_models else ollama_models_root()
        assert_local_ollama_model(args.model, models_root)

        work = Path(args.work_dir) if args.work_dir else ROOT / "runs" / "ollama-bridge"
        work.mkdir(parents=True, exist_ok=True)
        safe = args.model.replace("/", "_").replace(":", "_")
        gguf_path = work / f"{safe}.gguf"
        export_meta = export_weight_gguf(args.model, gguf_path, models_root)
        file_type = read_gguf_file_type(gguf_path)
        quant = gguf_quant_label(file_type)

        if args.hf_id:
            print(f"Downloading {args.hf_id} → {out} (preferred surgery base)", flush=True)
            snapshot_download_to(args.hf_id, out, revision=args.revision)
            import_mode = "hf_id"
            notes = (
                "Weights came from Hugging Face, not from dequantizing the Ollama GGUF. "
                "The exported blob digest is recorded for lineage only."
            )
        else:
            print(f"Dequantizing {gguf_path} ({quant}) → {out}", flush=True)
            gguf_to_hf(gguf_path, out, dtype=args.dtype)
            import_mode = "dequantized"
            notes = (
                "Weights were dequantized from the local Ollama GGUF blob. "
                "This is lossy when the blob is quantized; prefer --hf-id for surgery."
            )

        manifest = write_bridge_manifest(
            out,
            direction="ollama_to_hf",
            payload={
                "import_mode": import_mode,
                "source_ollama": args.model,
                "hf_id": args.hf_id,
                "hf_revision": args.revision,
                "blob_digest": export_meta["blob_digest"],
                "blob_bytes": export_meta["blob_bytes"],
                "gguf_path": export_meta["gguf_path"],
                "gguf_quant": quant,
                "gguf_file_type": file_type,
                "link_mode": export_meta["link_mode"],
                "ollama_template_present": bool(export_meta.get("template")),
                "ollama_system": export_meta.get("system"),
                "notes": notes,
            },
        )
        # Stash template alongside the checkpoint for hf_to_ollama to reuse.
        if export_meta.get("template"):
            (out / "ollama-template.txt").write_text(export_meta["template"])
        if export_meta.get("system"):
            (out / "ollama-system.txt").write_text(export_meta["system"] + "\n")

        print(f"Wrote checkpoint {out}", flush=True)
        print(f"Wrote provenance {manifest}", flush=True)
        print(f"import_mode={import_mode} gguf_quant={quant}", flush=True)
        return 0
    except BridgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
