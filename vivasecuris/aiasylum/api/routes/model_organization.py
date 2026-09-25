"""Names, notes, and experiment groups for models and recorded steps."""

from fastapi import APIRouter, HTTPException

from vivasecuris.aiasylum.api.model_organization import (
    ExperimentCreate,
    ExperimentUpdate,
    ItemUpdate,
    OrganizationUnavailable,
    get_store,
)

router = APIRouter()


def _call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except OrganizationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def get_organization():
    """Read the shared organization for this server's models and history."""
    return _call(get_store().read)


@router.post("/experiments", status_code=201)
def create_experiment(body: ExperimentCreate):
    return _call(get_store().create_experiment, **body.model_dump())


@router.patch("/experiments/{experiment_id}")
def update_experiment(experiment_id: str, body: ExperimentUpdate):
    return _call(get_store().update_experiment, experiment_id, **body.model_dump())


@router.put("/items")
def put_item(body: ItemUpdate):
    """Replace one item's editable metadata without modifying its provenance."""
    return _call(get_store().put_item, **body.model_dump())
