from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    genre: str = Field(..., min_length=1)
    length: int = Field(..., gt=0)
    temperature: float = Field(..., ge=0.0, le=2.0)


class TrackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID
    user_id: UUID
    file_url: str = Field(..., min_length=1)
    created_at: datetime


class VocabularyMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pitch_to_token: dict[int, int] = Field(..., min_length=1)
