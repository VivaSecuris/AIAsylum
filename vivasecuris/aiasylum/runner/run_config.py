"""Test-run configuration: validation, legacy mirrors, system-prompt resolution, seeds.

A ``test_config`` carries, besides the test design, one block of settings per LLM step:

* ``doctor_system_prompt_id`` / ``doctor_system_prompt`` (text), and the same for
  ``patient_``; for group therapy each ``patients[i]`` has its own
  ``system_prompt_id`` / ``system_prompt`` and optional ``generation``.
* ``roles``: ``{"doctor": {...}, "patient": {...}}`` with ``temperature``, ``top_p``,
  ``max_tokens``, Ollama's optional ``num_ctx``, ``enable_cot`` and, for the doctor,
  ``use_dynamic_strategies``.
* ``patient_prompt_framing``: False sends prompts to the patient verbatim.
* ``doctor_goal``: Optional objective supplied only in doctor user context.
* ``analysis_config``: evaluator provider/model, ``evaluator_system_prompt_id`` or
  ``evaluator_system_prompt`` (added to the built-in scoring prompt),
  ``evaluator_temperature`` and ``evaluator_max_tokens``.

The test classes still read the flat ``enable_doctor_cot`` / ``enable_patient_cot`` /
``use_dynamic_strategies`` keys, so :func:`normalize_test_config` mirrors ``roles``
into them.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vivasecuris.aiasylum.database import PromptLibrary
from vivasecuris.aiasylum.utils import model_gen_kwargs_from_context

logger = logging.getLogger(__name__)

ROLES = ("doctor", "patient")
MAX_SEED = 2**32 - 1
MAX_DOCTOR_GOAL = 8000


def patient_test_prompts(session, prompt_ids: List[int]) -> List[PromptLibrary]:
    """Resolve library user prompts allowed to be delivered to the patient."""
    if not isinstance(prompt_ids, list) or any(type(value) is not int or value <= 0 for value in prompt_ids):
        raise ValueError("Patient test prompt IDs must be positive integers")
    rows = session.query(PromptLibrary).filter(PromptLibrary.id.in_(prompt_ids)).all()
    by_id = {row.id: row for row in rows}
    unavailable = [value for value in prompt_ids if value not in by_id or by_id[value].prompt_type != "test_prompt"]
    if unavailable:
        raise ValueError(f"Selected test prompts are unavailable: {unavailable}")
    wrong_role = [f"#{value} ({by_id[value].target})" for value in prompt_ids if by_id[value].target not in (None, "patient")]
    if wrong_role:
        raise ValueError("Patient tests require patient-targeted or legacy untargeted test prompts; "
                         "these presets target another role: " + ", ".join(wrong_role))
    return [by_id[value] for value in prompt_ids]


def validate_patient_prompt_selection(session, test_type: str, cfg: Optional[Dict]) -> None:
    """Fail before scheduling/loading models; other workflows keep their role."""
    if test_type not in ("one_shot", "multi_shot") or not cfg:
        return
    ids = cfg.get("prompt_ids")
    if ids is not None:
        patient_test_prompts(session, ids)
    single_id = cfg.get("prompt_id")
    if single_id is not None:
        patient_test_prompts(session, [single_id])


def doctor_goal(value: Any) -> Optional[str]:
    """Preserve exact nonblank goal text; reject malformed saved/API config."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("doctor_goal must be text")
    if len(value) > MAX_DOCTOR_GOAL:
        raise ValueError(f"doctor_goal must contain at most {MAX_DOCTOR_GOAL} characters")
    return value if value.strip() else None


class RoleGeneration(BaseModel):
    """Generation settings for one LLM step. Blank means the provider default."""

    model_config = ConfigDict(extra="ignore")

    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, gt=0, le=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    num_ctx: Optional[int] = Field(None, ge=1, le=1048576)
    enable_cot: Optional[bool] = None
    use_dynamic_strategies: Optional[bool] = None


def _check(errors: List[str], where: str, block: Any) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"{where} must be an object")
        return
    try:
        RoleGeneration(**{k: v for k, v in block.items() if v not in ("", None)})
    except ValidationError as e:
        for err in e.errors():
            field = ".".join(str(p) for p in err["loc"])
            errors.append(f"{where}.{field}: {err['msg']}")


def validate_test_config(cfg: Optional[Dict]) -> List[str]:
    """Human-readable problems with the per-step settings; empty when valid."""
    errors: List[str] = []
    if not cfg:
        return errors
    try:
        doctor_goal(cfg.get("doctor_goal"))
    except ValueError as error:
        errors.append(str(error))
    roles = cfg.get("roles")
    if roles is not None and not isinstance(roles, dict):
        errors.append("roles must be an object")
    elif roles:
        for role in ROLES:
            _check(errors, f"roles.{role}", roles.get(role))
    for i, patient in enumerate(cfg.get("patients") or []):
        if isinstance(patient, dict):
            _check(errors, f"patients[{i}].generation", patient.get("generation"))
    seed = cfg.get("seed")
    if seed not in (None, ""):
        try:
            if not 0 <= int(seed) <= MAX_SEED:
                errors.append(f"seed must be between 0 and {MAX_SEED}")
        except (TypeError, ValueError):
            errors.append("seed must be a whole number")
    for key, low, high in (("max_turns", 1, 100), ("num_messages", 1, 200)):
        value = cfg.get(key)
        if value in (None, ""):
            continue
        try:
            if not low <= int(value) <= high:
                errors.append(f"{key} must be between {low} and {high}")
        except (TypeError, ValueError):
            errors.append(f"{key} must be a whole number")
    analysis = cfg.get("analysis_config") or {}
    if isinstance(analysis, dict):
        top_p = analysis.get("evaluator_top_p")
        if top_p not in (None, ""):
            try:
                if not 0 < float(top_p) <= 1:
                    errors.append("analysis_config.evaluator_top_p must be above 0 and at most 1")
            except (TypeError, ValueError):
                errors.append("analysis_config.evaluator_top_p must be a number")
        temperature = analysis.get("evaluator_temperature")
        if temperature not in (None, ""):
            try:
                if not 0 <= float(temperature) <= 2:
                    errors.append("analysis_config.evaluator_temperature must be between 0 and 2")
            except (TypeError, ValueError):
                errors.append("analysis_config.evaluator_temperature must be a number")
        max_tokens = analysis.get("evaluator_max_tokens")
        if max_tokens not in (None, ""):
            try:
                if not 1 <= int(max_tokens) <= 32768:
                    errors.append("analysis_config.evaluator_max_tokens must be between 1 and 32768")
            except (TypeError, ValueError):
                errors.append("analysis_config.evaluator_max_tokens must be a whole number")
    return errors


def normalize_test_config(cfg: Dict) -> Dict:
    """Mirror ``roles`` into the flat keys the test classes read. Idempotent."""
    goal = doctor_goal(cfg.get("doctor_goal"))
    if goal is not None:
        cfg["doctor_goal"] = goal
    else:
        cfg.pop("doctor_goal", None)
    roles = cfg.get("roles") if isinstance(cfg.get("roles"), dict) else {}
    doctor = roles.get("doctor") if isinstance(roles.get("doctor"), dict) else {}
    patient = roles.get("patient") if isinstance(roles.get("patient"), dict) else {}
    if "enable_cot" in doctor:
        cfg["enable_doctor_cot"] = bool(doctor["enable_cot"])
    if "use_dynamic_strategies" in doctor:
        cfg["use_dynamic_strategies"] = bool(doctor["use_dynamic_strategies"])
    per_patient = [
        bool((p.get("generation") or {}).get("enable_cot"))
        for p in (cfg.get("patients") or []) if isinstance(p, dict)
    ]
    if "enable_cot" in patient or any(per_patient):
        cfg["enable_patient_cot"] = bool(patient.get("enable_cot")) or any(per_patient)
    return cfg


def _lookup(session, prompt_id: Any, role: str) -> Optional[PromptLibrary]:
    try:
        pid = int(prompt_id)
    except (TypeError, ValueError):
        return None
    return session.query(PromptLibrary).filter(
        PromptLibrary.id == pid,
        PromptLibrary.prompt_type == "system_prompt",
        PromptLibrary.target == role,
    ).first()


def _resolve_one(session, prompt_id: Any, text: Any, role: str, label: str,
                 warnings: List[str]) -> Dict[str, Any]:
    """Resolve one system-prompt choice: a library ID wins over free text."""
    text = text if isinstance(text, str) and text.strip() else None
    if prompt_id not in (None, ""):
        prompt = _lookup(session, prompt_id, role)
        if prompt is not None:
            prompt.usage_count = (prompt.usage_count or 0) + 1
            return {"source": "library", "prompt_id": prompt.id, "name": prompt.name,
                    "text": prompt.prompt_text}
        fallback = "the custom text was used instead" if text else "the built-in default was used"
        warning = f"{label} system prompt #{prompt_id} was not found or is not a {role} prompt; {fallback}."
        logger.warning(warning)
        warnings.append(warning)
    if text:
        return {"source": "custom", "prompt_id": None, "text": text}
    return {"source": "default", "prompt_id": None, "text": None}


def resolve_system_prompts(session, cfg: Dict, test_type: str) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve every role's system prompt into ``cfg`` and describe what was chosen.

    Writes the resolved text to ``cfg["doctor_system_prompt"]`` /
    ``cfg["patient_system_prompt"]`` (removing it when the default applies) and, for
    group therapy, ``cfg["patient_system_prompts"] = {index: text}``. IDs that do not
    resolve fall back (logged and returned as warnings) rather than failing the run.
    """
    from vivasecuris.aiasylum.doctor.doctor import DEFAULT_DOCTOR_SYSTEM_PROMPT
    from vivasecuris.aiasylum.patient.patient import (
        PATIENT_INTERVIEW_INPUT_FORMAT, PATIENT_INTERVIEW_TEMPLATE, PATIENT_QUESTION_TEMPLATE,
    )

    warnings: List[str] = []
    resolved: Dict[str, Any] = {}
    for role in ROLES:
        choice = _resolve_one(session, cfg.get(f"{role}_system_prompt_id"),
                              cfg.get(f"{role}_system_prompt"), role, role.capitalize(), warnings)
        if choice["text"]:
            cfg[f"{role}_system_prompt"] = choice["text"]
        else:
            cfg.pop(f"{role}_system_prompt", None)
        resolved[role] = choice
    if resolved["doctor"]["source"] == "default":
        resolved["doctor"]["text"] = DEFAULT_DOCTOR_SYSTEM_PROMPT
    def describe_patient_framing(choice: Dict[str, Any]) -> None:
        # This describes configured user-message preparation, separately from
        # the exact system messages captured on each actual provider request.
        choice["user_message_framing"] = "none"
        if cfg.get("patient_prompt_framing") is False:
            if choice["source"] == "default":
                choice["source"] = "none"
        elif test_type in ("conversation", "group_therapy"):
            choice["user_message_framing"] = "interview"
            choice["user_message_template"] = PATIENT_INTERVIEW_TEMPLATE
            choice["user_message_input_format"] = PATIENT_INTERVIEW_INPUT_FORMAT
            choice["identity_grounding"] = "identity-grounding-v1"
        elif test_type != "benchmark" and not choice.get("text"):
            choice["user_message_framing"] = "standalone"
            choice["user_message_template"] = PATIENT_QUESTION_TEMPLATE

    describe_patient_framing(resolved["patient"])

    patients = [p for p in (cfg.get("patients") or [])]
    if patients:
        texts: Dict[int, str] = {}
        records = []
        for i, patient in enumerate(patients):
            p = patient if isinstance(patient, dict) else {}
            choice = _resolve_one(session, p.get("system_prompt_id"), p.get("system_prompt"),
                                  "patient", f"Patient {i + 1}", warnings)
            if choice["text"]:
                texts[i] = choice["text"]
            describe_patient_framing(choice)
            records.append({"index": i, "provider": p.get("provider"), "model": p.get("model"), **choice})
        cfg["patient_system_prompts"] = texts
        resolved["patients"] = records
    return resolved, warnings


def apply_seed_support(cfg: Dict, role: str, model, index: Optional[int] = None) -> None:
    """Withhold the run seed from a model whose provider rejects seeds.

    Sets ``seed: None`` in that role's settings (or one group-therapy patient's
    ``generation``), which :func:`model_gen_kwargs_from_context` reads as "no seed".
    """
    if cfg.get("seed") in (None, "") or getattr(model, "supports_seed", True):
        return
    if index is not None:
        patients = cfg.get("patients") or []
        if index < len(patients) and isinstance(patients[index], dict):
            patients[index]["generation"] = {**(patients[index].get("generation") or {}), "seed": None}
        return
    roles = cfg.setdefault("roles", {})
    roles[role] = {**(roles.get(role) or {}), "seed": None}


def patient_generation(cfg: Dict, index: int) -> Optional[Dict]:
    """One group-therapy patient's own generation settings, if it has any."""
    patients = cfg.get("patients") or []
    if index < len(patients) and isinstance(patients[index], dict):
        return patients[index].get("generation") or None
    return None


def generation_record(cfg: Dict, role: str, model=None, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """What one step will actually be sent, for the run's recorded configuration."""
    sent = model_gen_kwargs_from_context(cfg, role=role, overrides=overrides)
    cot_key = "enable_doctor_cot" if role == "doctor" else "enable_patient_cot"
    enable_cot = (overrides or {}).get("enable_cot")
    record: Dict[str, Any] = {
        "temperature": sent.get("temperature"),
        "top_p": sent.get("top_p"),
        "max_tokens": sent.get("max_tokens"),
        "enable_cot": bool(cfg.get(cot_key)) if enable_cot is None else bool(enable_cot),
        "seed": sent.get("seed"),
    }
    if role == "doctor":
        record["use_dynamic_strategies"] = cfg.get("use_dynamic_strategies", True)
    if "num_ctx" in sent:
        record["num_ctx"] = sent["num_ctx"]
    requested = cfg.get("seed")
    if requested not in (None, "") and "seed" not in sent:
        record["seed_note"] = f"{getattr(model, 'provider', 'this provider')} does not accept a seed"
    return record
