"""One clear entry point for paired local-model benchmark comparisons."""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from vivasecuris.aiasylum.api import benchmark_campaigns as campaigns

router = APIRouter()


class CampaignRequest(BaseModel):
    name: str = Field(default="Model comparison", min_length=1, max_length=120)
    models: list[str] = Field(min_length=1, max_length=30)
    benchmarks: list[str] = Field(min_length=1, max_length=4)
    num_samples: int = Field(default=25, ge=1, le=200)
    seed: int = Field(default=0, ge=0, le=2147483647)
    max_new_tokens: int = Field(default=256, ge=16, le=2048)

    @field_validator("models", "benchmarks")
    @classmethod
    def unique_nonempty(cls, values):
        if any(not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError("Selections must be nonempty and unique")
        return values

    @field_validator("benchmarks")
    @classmethod
    def supported(cls, values):
        if any(v not in campaigns.SUPPORTED for v in values):
            raise ValueError(f"Supported comparison checks: {', '.join(campaigns.SUPPORTED)}")
        return values


@router.get("")
async def list_campaigns():
    return {"campaigns": campaigns.list_campaigns()}


@router.post("", status_code=201)
async def create_campaign(request: CampaignRequest):
    try:
        resolved, revisions = await asyncio.to_thread(campaigns.pin_inputs, request.models, request.benchmarks)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Could not pin benchmark inputs: {exc}") from exc
    campaign = campaigns.create_campaign(request.model_dump(), resolved, revisions)
    campaigns.schedule(campaign["id"])
    return campaign


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str):
    value = campaigns.get_campaign(campaign_id)
    if value is None:
        raise HTTPException(404, "Benchmark comparison not found")
    return value


@router.post("/{campaign_id}/cancel")
async def cancel_campaign(campaign_id: str):
    value = campaigns.cancel_campaign(campaign_id)
    if value is None:
        raise HTTPException(404, "Benchmark comparison not found")
    return value


@router.post("/{campaign_id}/rescore", status_code=201)
async def rescore_campaign(campaign_id: str):
    from vivasecuris.aiasylum.api.benchmark_rescoring import rescore_campaign as rescore, RescoreConflict
    try:
        return await asyncio.to_thread(rescore, campaign_id)
    except KeyError as exc:
        raise HTTPException(404, "Benchmark comparison not found") from exc
    except RescoreConflict as exc:
        raise HTTPException(409, str(exc)) from exc
