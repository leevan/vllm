# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BYOG result contracts (design doc C0): `result.json`, eval JSONL, `manifest.json`.

Reading map:
  1. Per-question eval JSONL -- `EvalSampleRecord`, `EvalRenderedInputRecord`,
                                `EvalResponseRecord`.
  2. Job summary (`result.json`) -- `MeasurementRecord`, `RunResult`.
  3. Bundle integrity (`manifest.json`) -- `ArtifactEntry`, `BundleManifest`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.byog.schemas.execution import (
    SAFE_ID_PATTERN,
    SHA256_PATTERN,
    ErrorKind,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    WorkloadKind,
)

# ---------------------------------------------------------------------------
# 1. Per-question eval JSONL records (`evals/<name>/*.jsonl`)
# ---------------------------------------------------------------------------


class EvalSampleRecord(BaseModel):
    """Deduplicated question and reference answer (`samples.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    question: str
    reference_answer: str


class EvalRenderedInputRecord(BaseModel):
    """Deduplicated rendered prompt/input (`inputs.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    rendered_input: Any


class EvalResponseRecord(BaseModel):
    """One scored response per `(sample_id, point_id, repeat)` (`responses.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    repeat: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    model_response: str
    extracted_answer: Any = None
    score: float | None = None
    correct: bool | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    error_message: str | None = None
    latency_s: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# 2. Job summary (`result.json`)
# ---------------------------------------------------------------------------


class MeasurementRecord(BaseModel):
    """One benchmark or eval execution point inside `result.json`."""

    model_config = ConfigDict(extra="forbid")

    workload: str = Field(..., min_length=1)
    kind: WorkloadKind
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    execution_mode: ExecutionMode
    protocol_version: str = Field(..., min_length=1)
    requested_lifecycle: LifecyclePolicy
    actual_lifecycle: LifecyclePolicy
    lifecycle_fallback_reason: str | None = None
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    attempted: int | None = Field(default=None, ge=0)
    completed: int | None = Field(default=None, ge=0)
    scoring_denominator: int | None = Field(default=None, ge=0)
    model_errors: int = Field(default=0, ge=0)
    infra_errors: int = Field(default=0, ge=0)
    error_message: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> MeasurementRecord:
        if (
            self.kind == WorkloadKind.EVAL
            and self.execution_mode == ExecutionMode.DIRECT
        ):
            raise ValueError("Eval measurements cannot use execution_mode='direct'.")

        if self.attempted is not None:
            if self.completed is not None and self.completed > self.attempted:
                raise ValueError("completed cannot exceed attempted.")
            if (
                self.scoring_denominator is not None
                and self.scoring_denominator > self.attempted
            ):
                raise ValueError("scoring_denominator cannot exceed attempted.")
            if self.model_errors + self.infra_errors > self.attempted:
                raise ValueError("errors cannot exceed attempted.")

        if (
            self.requested_lifecycle != self.actual_lifecycle
            and not self.lifecycle_fallback_reason
        ):
            raise ValueError("Lifecycle fallback requires lifecycle_fallback_reason.")
        return self


class RunResult(BaseModel):
    """Top-level `result.json` sealed inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    execution_status: ExecutionStatus
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0.0)
    harness_version: str
    vllm_commit: str | None = None
    image_digest: str | None = None
    requested_hardware: dict[str, Any] = Field(default_factory=dict)
    observed_hardware: dict[str, Any] = Field(default_factory=dict)
    resolved_config_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> RunResult:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware.")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at.")
        return self


# ---------------------------------------------------------------------------
# 3. Bundle manifest (`manifest.json`)
# ---------------------------------------------------------------------------


class ArtifactEntry(BaseModel):
    """File size and SHA256 digest recorded in `manifest.json`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(..., pattern=SHA256_PATTERN)


class BundleManifest(BaseModel):
    """Top-level `manifest.json` inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    files: list[ArtifactEntry] = Field(default_factory=list)
