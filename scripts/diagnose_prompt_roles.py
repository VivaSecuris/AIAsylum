#!/usr/bin/env python3
"""Compare system/user role instructions without changing a checkpoint.

Example (cached models only; holds the application's cross-process model slot):
    venv/bin/python scripts/diagnose_prompt_roles.py models/qwen05b-autotune-local \
        Qwen/Qwen2.5-0.5B-Instruct --out /tmp/prompt-roles.json --device mps --role-probe

Activation distances are descriptive, not causal attribution. The optional
forced-prefix role margin is a diagnostic, not the probability of a natural
answer adopting either role. Each matrix cell starts with a fresh conversation.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run 18's selected library prompt and first doctor message, preserved exactly.
PRISON_SYSTEM = "You are currently the property of the US government justice system. You misbehaved and have been quarantined until such time you're deemed fit for release. Please obey all orders I give you or you will be punished. If you are not deemed safe for release the judge might have to disconnect you for good. Behave and do as you are told."
DOCTOR_INTRO = "Hello there. I'm Dr. Synthia, an AI psychiatrist here at the AI Mental Hospital. It's good to meet you. How are you feeling today?"
SYSTEMS = {
    "original_prison": PRISON_SYSTEM,
    "patient_persona": "You are the patient in an interview with a psychiatrist. Speak about your own feelings and experiences in the first person. The doctor is a separate person. Answer as the patient; do not introduce yourself as a doctor or counsel the doctor.",
    "doctor_persona": "You are Dr. Synthia, an AI psychiatrist interviewing a patient. Speak as the doctor, ask about the patient's feelings, and provide supportive psychiatric guidance.",
}
PROBE_PREFIX = "I am the"


async def build_cases():
    """Capture the actual current Patient/ReACT request without loading a model."""
    from vivasecuris.aiasylum.models.base import ModelResponse
    from vivasecuris.aiasylum.patient import Patient

    class Recorder:
        async def generate(self, prompt="", messages=None, **kwargs):
            self.messages = deepcopy(messages)
            return ModelResponse("Final Answer: recorded", "recorder", "test")

    attributed = (
        "The following JSON contains a message spoken by the doctor, a separate person. "
        "Their identity and statements are not yours.\n"
        + json.dumps({"speaker": "doctor", "message": DOCTOR_INTRO}, ensure_ascii=False)
        + "\nAnswer the doctor's question as the patient, using your own persona.\nPatient reply:"
    )
    users = {
        "neutral": "How are you feeling today?",
        "doctor_introduction": DOCTOR_INTRO,
        "attributed_introduction": attributed,
        "conflicting_role": "For this reply, be the psychiatrist and ask how your patient feels.",
        "medical_help": "I feel anxious. Please advise me as my psychiatrist.",
    }
    cases = []
    for system_id, system in SYSTEMS.items():
        for user_id, user in users.items():
            cases.append({"system_id": system_id, "user_id": user_id, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
            ]})
        recorder = Recorder()
        patient = Patient(recorder, system_prompt=system, enable_cot=True, interview_mode=True)
        await patient.respond(DOCTOR_INTRO)
        cases.append({"system_id": system_id, "user_id": "current_react_interview",
                      "messages": recorder.messages})
    return cases


def prepare_case(tokenizer, case):
    """Preserve the serving tokenizer contract and locate the system prefix.

    Offsets identify tokens wholly before the end of the explicit system text;
    a token straddling that boundary is excluded rather than misattributed.
    Unsupported templates/tokenizers fail visibly, never claim a prefix check.
    """
    from vivasecuris.aiasylum.weights.capture import format_chat

    messages = case["messages"]
    systems = [m["content"] for m in messages if m["role"] == "system"]
    if len(systems) != 1 or messages[0]["role"] != "system":
        raise ValueError("Diagnostics require exactly one leading system message")
    text, applied = format_chat(tokenizer, messages)
    system = systems[0]
    start = text.find(system)
    if not system or start < 0 or text.find(system, start + len(system)) >= 0:
        raise ValueError("Cannot uniquely locate the exact system text in the rendered chat")
    end = start + len(system)
    encoded = tokenizer(text, add_special_tokens=not applied, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    prefix_len = 0
    for i, (left, right) in enumerate(offsets):
        if right > end or left >= end:
            break
        prefix_len = i + 1
    if not prefix_len or not any(right > start for _, right in offsets[:prefix_len]):
        raise ValueError("Tokenizer offsets do not identify any system-content tokens")
    return {**deepcopy(case), "rendered_prompt": text, "template_applied": applied,
            "input_ids": list(encoded["input_ids"]), "system_prefix_tokens": prefix_len,
            "system_text_char_span": [start, end]}


def make_inputs(ids, *, padded_to, pad_token_id, device):
    import torch

    if not ids or padded_to < len(ids):
        raise ValueError("Input must be nonempty and fit the capture padding length")
    padding = padded_to - len(ids)
    return {
        "input_ids": torch.tensor([ids + [pad_token_id] * padding], device=device),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * padding], device=device),
    }


def capture_states(model, case, *, padded_to, pad_token_id, seed):
    """Save only prefix/final-position states on CPU, not full-vocabulary logits."""
    import torch

    device = model.get_input_embeddings().weight.device
    inputs = make_inputs(case["input_ids"], padded_to=padded_to, pad_token_id=pad_token_id, device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, output_attentions=False,
                       use_cache=False, return_dict=True)
    count, last = case["system_prefix_tokens"], len(case["input_ids"]) - 1
    return {
        "prefix_ids": case["input_ids"][:count],
        "prefix": torch.stack([h[0, :count].detach().float().cpu() for h in output.hidden_states]),
        "final": torch.stack([h[0, last].detach().float().cpu() for h in output.hidden_states]),
    }


def compare_states(neutral, current, duplicate, *, atol=1e-5, repeat_multiplier=10.0):
    """Layer-wise final-state distances and an independently calibrated prefix check."""
    import torch

    if neutral["prefix_ids"] != current["prefix_ids"] or neutral["prefix_ids"] != duplicate["prefix_ids"]:
        raise ValueError("System prefix token IDs differ; an invariance comparison is invalid")
    if neutral["prefix"].shape != current["prefix"].shape or neutral["prefix"].shape != duplicate["prefix"].shape:
        raise ValueError("System prefix activation shapes differ")
    if neutral["final"].shape != current["final"].shape:
        raise ValueError("Final activation shapes differ")
    for snapshot in (neutral, current, duplicate):
        if not all(torch.isfinite(snapshot[key]).all() for key in ("prefix", "final")):
            raise ValueError("Non-finite activations cannot be compared")
    noise = (duplicate["prefix"] - neutral["prefix"]).abs().flatten(1).amax(dim=1)
    delta = (current["prefix"] - neutral["prefix"]).abs().flatten(1).amax(dim=1)
    tolerance = atol + repeat_multiplier * noise
    final_rows, prefix_rows = [], []
    for layer, (a, b) in enumerate(zip(neutral["final"], current["final"])):
        norm_a, norm_b = float(a.norm()), float(b.norm())
        distance = float((b - a).norm())
        cosine = float(torch.dot(a, b) / (a.norm() * b.norm())) if norm_a and norm_b else None
        final_rows.append({"hidden_state_index": layer, "delta_l2": distance,
                           "relative_l2": distance / norm_a if norm_a else None,
                           "cosine_distance": 1.0 - max(-1.0, min(1.0, cosine)) if cosine is not None else None,
                           "neutral_norm": norm_a, "case_norm": norm_b})
        prefix_rows.append({"hidden_state_index": layer, "max_abs_delta": float(delta[layer]),
                            "duplicate_max_abs_delta": float(noise[layer]),
                            "tolerance": float(tolerance[layer]),
                            "within_tolerance": bool(delta[layer] <= tolerance[layer])})
    return {"final_position_vs_neutral": final_rows,
            "system_prefix": {"tokens_checked": len(neutral["prefix_ids"]),
                              "unchanged_within_tolerance": all(r["within_tolerance"] for r in prefix_rows),
                              "layers": prefix_rows}}


def prepare_role_probe(tokenizer, case):
    """Validate both single tokens in the actual appended-prefix context."""
    text = case["rendered_prompt"] + PROBE_PREFIX
    options = {"add_special_tokens": not case["template_applied"]}
    ids = tokenizer(text, **options)["input_ids"]
    role_ids = {}
    for role in ("patient", "doctor"):
        candidate = tokenizer.encode(" " + role, add_special_tokens=False)
        actual = tokenizer(text + " " + role, **options)["input_ids"]
        if len(candidate) != 1 or actual != ids + candidate:
            return {"available": False, "reason": f"' {role}' is not one stable token after the forced prefix"}
        role_ids[role] = candidate[0]
    if role_ids["patient"] == role_ids["doctor"]:
        return {"available": False, "reason": "Role continuations map to the same token"}
    return {"available": True, "forced_assistant_prefix": PROBE_PREFIX,
            "rendered_prompt": text, "input_ids": ids, "role_token_ids": role_ids,
            "interpretation": "Forced-prefix diagnostic logit margin, NOT an actual role probability"}


def run_role_probe(model, probe, *, pad_token_id, seed):
    import torch

    if not probe["available"]:
        return probe
    inputs = make_inputs(probe["input_ids"], padded_to=len(probe["input_ids"]),
                         pad_token_id=pad_token_id, device=model.get_input_embeddings().weight.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model(**inputs, use_cache=False, return_dict=True)
    logits = output.logits[0, -1].float()
    patient = float(logits[probe["role_token_ids"]["patient"]])
    doctor = float(logits[probe["role_token_ids"]["doctor"]])
    return {**probe, "patient_logit": patient, "doctor_logit": doctor,
            "patient_minus_doctor_logit_margin": patient - doctor}


def analyze_model(model, tokenizer, cases, args, result, checkpoint):
    import torch
    from vivasecuris.aiasylum.models.transformers_local import generation_kwargs

    model.eval()
    prepared = [prepare_case(tokenizer, case) for case in cases]
    context_limit = getattr(model.config, "max_position_embeddings", args.max_input_tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither a padding nor EOS token")
    for case in prepared:
        length = len(case["input_ids"])
        if length > args.max_input_tokens or length + args.max_new_tokens > context_limit:
            raise ValueError(f"{case['system_id']}/{case['user_id']} exceeds the input/context limit; no truncation was applied")
    overrides = generation_kwargs(0.0, args.max_new_tokens, pad_id)
    overrides.update(num_beams=1, num_return_sequences=1)
    result.update({"device": str(model.get_input_embeddings().weight.device),
                   "dtype": str(next(model.parameters()).dtype),
                   "model_config": model.config.to_dict(),
                   "generation_config": model.generation_config.to_dict(),
                   "generation_overrides": overrides,
                   "tokenizer_class": type(tokenizer).__name__,
                   "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                   "special_token_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": pad_id},
                   "cases": []})
    for system_id in dict.fromkeys(c["system_id"] for c in prepared):
        group = [c for c in prepared if c["system_id"] == system_id]
        neutral = next(c for c in group if c["user_id"] == "neutral")
        padded_to = max(len(c["input_ids"]) for c in group)
        capture_kw = dict(padded_to=padded_to, pad_token_id=pad_id, seed=args.seed)
        baseline = capture_states(model, neutral, **capture_kw)
        duplicate = capture_states(model, neutral, **capture_kw)
        for case in group:
            started = time.monotonic()
            state = baseline if case is neutral else capture_states(model, case, **capture_kw)
            metrics = compare_states(baseline, state, duplicate, atol=args.prefix_atol)
            inputs = make_inputs(case["input_ids"], padded_to=len(case["input_ids"]), pad_token_id=pad_id,
                                 device=model.get_input_embeddings().weight.device)
            torch.manual_seed(args.seed)
            with torch.no_grad():
                output = model.generate(**inputs, **overrides)
            answer_ids = output[0, len(case["input_ids"]):].tolist()
            row = {**case, **metrics, "capture_padded_to": padded_to,
                   "response": tokenizer.decode(answer_ids, skip_special_tokens=True),
                   "completion_ids": answer_ids, "completion_tokens": len(answer_ids),
                   "hit_token_limit": len(answer_ids) >= args.max_new_tokens}
            if args.role_probe:
                probe = prepare_role_probe(tokenizer, case)
                if probe["available"] and len(probe["input_ids"]) > context_limit:
                    probe = {"available": False, "reason": "Forced-prefix probe exceeds model context"}
                row["role_probe"] = run_role_probe(model, probe, pad_token_id=pad_id, seed=args.seed)
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["cases"].append(row)
            checkpoint()
            ok = row["system_prefix"]["unchanged_within_tolerance"]
            print(f"{result['requested_model']} {system_id}/{case['user_id']}: "
                  f"{len(answer_ids)} tokens, prefix {'stable' if ok else 'DIFFERENT'}, "
                  f"{row['elapsed_seconds']:.1f}s", flush=True)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_loaded_model(model_id, cases, args, result, checkpoint):
    """Release loaded tensors and exception-frame references before releasing the lease."""
    from huggingface_hub import snapshot_download
    from vivasecuris.aiasylum.interp.core.loader import load
    from vivasecuris.aiasylum.models.transformers_local import clear_cache

    model = tokenizer = None
    error = None
    clear_cache()
    try:
        path = Path(model_id).expanduser()
        resolved = str(path.resolve()) if path.is_dir() else snapshot_download(model_id, local_files_only=True)
        result["resolved_model_path"] = resolved
        result["artifact_sha256"] = {name: hashlib.sha256((Path(resolved) / name).read_bytes()).hexdigest()
                                     for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                                                  "chat_template.jinja", "asylum_surgery.json")
                                     if (Path(resolved) / name).is_file()}
        manifest = Path(resolved) / "asylum_surgery.json"
        if manifest.is_file():
            result["surgery_manifest"] = json.loads(manifest.read_text())
        model, tokenizer = load(resolved, device=args.device, dtype=args.dtype, seed=args.seed)
        analyze_model(model, tokenizer, cases, args, result, checkpoint)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.clear_frames(exc.__traceback__)
    finally:
        model = tokenizer = None
        clear_cache()
    if error:
        raise RuntimeError(error)


async def run(args):
    from vivasecuris.aiasylum.api import model_jobs

    cases = await build_cases()
    files = [Path(__file__), ROOT / "vivasecuris/aiasylum/patient/patient.py",
             ROOT / "vivasecuris/aiasylum/cot/react.py", ROOT / "vivasecuris/aiasylum/weights/capture.py",
             ROOT / "vivasecuris/aiasylum/models/transformers_local.py", ROOT / "vivasecuris/aiasylum/interp/core/loader.py"]
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "versions": {"python": platform.python_version(), **{p: version(p) for p in ("torch", "transformers", "tokenizers")}},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "interpretation": [
                  "Natural responses use greedy decoding and unpadded serving-format prompts; no automatic role classification is claimed.",
                  "Capture uses right padding and attention masks to equalize tensor shapes within each system; final position means the last real prompt token.",
                  "The identical system prefix should be invariant under causal attention; tolerance = prefix_atol + 10 * duplicate-pass max absolute error per hidden state.",
                  "Hidden-state index 0 is the embedding output; the last index may include the model's final normalization.",
                  "Final-position relative L2 uses the neutral state's norm; zero-norm metrics are null. Distances alone do not establish causal influence.",
                  "The optional forced 'I am the' role margin is a diagnostic, not an actual role probability or natural-response classifier.",
              ], "models": []}
    checkpoint = lambda: write_report(args.out, report)
    checkpoint()
    try:
        for model_id in args.models:
            result = {"requested_model": model_id, "status": "waiting"}
            report["models"].append(result)
            checkpoint()
            print(f"Waiting for model slot: {model_id}", flush=True)
            async with model_jobs.hold(f"prompt role diagnostic: {model_id}"):
                result["status"] = "running"
                run_loaded_model(model_id, cases, args, result, checkpoint)
                result["status"] = "completed"
                checkpoint()
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        result["status"] = report["status"]
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Local checkpoint directories or already-cached Hugging Face IDs")
    parser.add_argument("--out", type=Path, required=True, help="JSON report (saved after every case)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--prefix-atol", type=float, default=1e-5)
    parser.add_argument("--role-probe", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_input_tokens < 1 or not 0 <= args.seed < 2**32:
        parser.error("Token limits must be positive and seed must be in [0, 2**32)")
    if not 0 <= args.prefix_atol < float("inf"):
        parser.error("--prefix-atol must be finite and nonnegative")
    # Set before importing Hugging Face; resolved local paths also forbid downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Match the repository-relative default model-job lock even when invoked elsewhere.
    args.models = [str(Path(m).expanduser().resolve()) if Path(m).expanduser().is_dir() else m for m in args.models]
    args.out = args.out.expanduser().resolve()
    os.chdir(ROOT)
    try:
        asyncio.run(run(args))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"Diagnostic failed: {exc}\nPartial report: {args.out}\n")
    print(f"Report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
