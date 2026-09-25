"""Editable experiment labels and membership, separate from recorded provenance.

The small workspace document is read again for every update. A stable lock file
serializes writers across API processes; replacing the JSON file atomically
keeps a failed write from erasing the previous organization.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


MAX_EXPERIMENTS = 100
MAX_ITEMS = 1000
MAX_STORE_BYTES = 32 * 1024 * 1024
_thread_lock = threading.RLock()

Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=120)]
Label = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, max_length=120)]
Notes = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, max_length=4000)]
ItemKey = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1024)]
ExperimentId = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=36)]


class OrganizationUnavailable(RuntimeError):
    """The organization document cannot safely be read or written."""


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExperimentCreate(_Schema):
    name: Name
    notes: Notes = ""


class ExperimentUpdate(_Schema):
    name: Name | None = None
    notes: Notes | None = None


class Experiment(ExperimentCreate):
    id: ExperimentId
    created_at: str

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if str(UUID(value)) != value:
            raise ValueError("experiment ID must be a canonical UUID")
        return value

    @field_validator("created_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class ItemMetadata(_Schema):
    label: Label = ""
    notes: Notes = ""
    experiment_ids: list[ExperimentId] = Field(default_factory=list, max_length=MAX_EXPERIMENTS)

    @field_validator("experiment_ids")
    @classmethod
    def unique_assignments(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("experiment_ids must contain unique IDs")
        return value


def _validate_key(key: str) -> str:
    prefix, separator, reference = key.partition(":")
    if not separator or prefix not in {"model", "step"} or not reference.strip():
        raise ValueError("item key must be model:{model_ref} or step:{lineage_node_id}")
    if key != key.strip() or any(ord(character) < 32 for character in key):
        raise ValueError("item key must not contain surrounding whitespace or control characters")
    return key


class ItemUpdate(ItemMetadata):
    key: ItemKey

    @field_validator("key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        return _validate_key(value)


class Organization(_Schema):
    experiments: list[Experiment] = Field(max_length=MAX_EXPERIMENTS)
    items: dict[ItemKey, ItemMetadata] = Field(max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def valid_memberships(self) -> Organization:
        ids = {experiment.id for experiment in self.experiments}
        if len(ids) != len(self.experiments):
            raise ValueError("experiment IDs must be unique")
        for key, metadata in self.items.items():
            _validate_key(key)
            if not set(metadata.experiment_ids).issubset(ids):
                raise ValueError("experiment_ids contains an unknown experiment")
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


class OrganizationStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _locked(self):
        # Lock a separate inode: the data file itself is replaced on every save.
        with _thread_lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                with os.fdopen(descriptor, "a") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise OrganizationUnavailable(
                    "Model organization storage is unavailable; no organization changes were saved."
                ) from exc

    def _read(self) -> Organization:
        try:
            with self.path.open("rb") as source:
                raw = source.read(MAX_STORE_BYTES + 1)
        except FileNotFoundError:
            return Organization(experiments=[], items={})
        try:
            if len(raw) > MAX_STORE_BYTES:
                raise ValueError("organization file exceeds the size limit")
            document = json.loads(raw, object_pairs_hook=_unique_object)
            return Organization.model_validate(document)
        except (ValueError, TypeError, RecursionError) as exc:
            raise OrganizationUnavailable(
                "Model organization storage is invalid. Restore runs/model-organization.json "
                "from a valid backup before editing; the existing file was preserved."
            ) from exc

    def _write(self, organization: Organization) -> None:
        # Validate again before touching disk, including collection limits.
        document = Organization.model_validate(organization.model_dump()).model_dump()
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".model-organization-", suffix=".json", dir=self.path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                json.dump(document, target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def read(self) -> dict:
        with self._locked():
            return self._read().model_dump()

    def create_experiment(self, name: str, notes: str = "") -> dict:
        request = ExperimentCreate(name=name, notes=notes)
        with self._locked():
            organization = self._read()
            if len(organization.experiments) >= MAX_EXPERIMENTS:
                raise ValueError(f"A workspace supports at most {MAX_EXPERIMENTS} experiments.")
            experiment = Experiment(
                id=str(uuid4()), name=request.name, notes=request.notes,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            organization.experiments.append(experiment)
            self._write(organization)
            return experiment.model_dump()

    def update_experiment(self, experiment_id: str, *, name: str | None = None, notes: str | None = None) -> dict:
        request = ExperimentUpdate(name=name, notes=notes)
        with self._locked():
            organization = self._read()
            experiment = next((item for item in organization.experiments if item.id == experiment_id), None)
            if experiment is None:
                raise LookupError("Experiment not found.")
            if request.name is not None:
                experiment.name = request.name
            if request.notes is not None:
                experiment.notes = request.notes
            self._write(organization)
            return experiment.model_dump()

    def put_item(self, key: str, label: str = "", notes: str = "", experiment_ids: list[str] | None = None) -> dict:
        request = ItemUpdate(key=key, label=label, notes=notes,
                             experiment_ids=[] if experiment_ids is None else experiment_ids)
        with self._locked():
            organization = self._read()
            known_ids = {experiment.id for experiment in organization.experiments}
            if not set(request.experiment_ids).issubset(known_ids):
                raise ValueError("experiment_ids contains an unknown experiment.")
            if not (request.label or request.notes or request.experiment_ids):
                organization.items.pop(request.key, None)
            else:
                if request.key not in organization.items and len(organization.items) >= MAX_ITEMS:
                    raise ValueError(f"A workspace supports at most {MAX_ITEMS} organized models and steps.")
                organization.items[request.key] = ItemMetadata(**request.model_dump(exclude={"key"}))
            self._write(organization)
            return organization.model_dump()


def get_store() -> OrganizationStore:
    from config import settings

    return OrganizationStore(settings.project_root / "runs" / "model-organization.json")
