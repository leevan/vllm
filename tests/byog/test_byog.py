# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 verification suite (schemas, resolver, bundle, executor, CLI, E2E)."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.byog.fake_launcher import FakeLauncher, assert_plugin_contract
from tools.byog.bundle import (
    BundleValidationError,
    DeduplicatedEvalRecordWriter,
    seal_bundle,
    validate_bundle,
)
from tools.byog.catalog import Catalog
from tools.byog.cli import EXIT_OK, main
from tools.byog.executor.coordinator import Coordinator
from tools.byog.interfaces import WorkloadPoint
from tools.byog.schemas.execution import (
    EvalSelection,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    ResolvedJob,
    RunSpec,
    SuiteDefinition,
    WorkloadKind,
)
from tools.byog.schemas.results import MeasurementRecord
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def _gsm8k_overrides() -> dict[str, object]:
    return {
        "train_source": str(FIXTURES / "gsm8k_mini_train.jsonl"),
        "test_source": str(FIXTURES / "gsm8k_mini_test.jsonl"),
        "num_questions": 6,
    }


def test_schema_invariants() -> None:
    # Unsafe point_id, invalid concurrency, invalid hash, invalid counts/fallback.
    with pytest.raises(ValueError):
        WorkloadPoint("../../escape")
    with pytest.raises(ValidationError):
        EvalSelection(name="gsm8k", concurrency=[0, -1])
    with pytest.raises(ValidationError):
        SuiteDefinition.model_validate(
            {"name": "bad", "evals": [{"name": "gsm8k", "concurrency": [0]}]}
        )
    with pytest.raises(ValidationError):
        ResolvedJob(resolved_config_hash="not-a-sha256")
    # ResolvedJob exists before any run (attempt_id optional).
    assert ResolvedJob().attempt_id is None

    # Fallback without reason is rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="http_smoke",
            kind=WorkloadKind.BENCHMARK,
            point_id="c1",
            execution_mode=ExecutionMode.SERVER,
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.RESET_CACHE,
            actual_lifecycle=LifecyclePolicy.RESTART_SERVER,
        )
    # Completed > attempted is rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="http_smoke",
            kind=WorkloadKind.BENCHMARK,
            point_id="c1",
            execution_mode=ExecutionMode.SERVER,
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.NONE,
            actual_lifecycle=LifecyclePolicy.NONE,
            attempted=2,
            completed=3,
        )


@pytest.mark.parametrize(
    "bad_override",
    [
        {"recipe": {"model": "org/m"}},
        {"matrix": {"tp": [1, 2]}},
        {"compare_baseline": True},
        {"hardware": {"nodes": 2}},
    ],
)
def test_runspec_rejects_phase1_fields(bad_override: dict) -> None:
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "benchmarks": [{"type": "http_smoke"}],
    }
    base.update(bad_override)
    with pytest.raises(ValidationError):
        RunSpec.model_validate(base)


@pytest.mark.parametrize(
    "bad_override",
    [
        {"serve": {"config": {"model": "m", "revision": "main"}}},
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"HF_TOKEN": "x"},
            }
        },
        {"benchmarks": [{"type": "http_smoke", "unknown_opt": 1}]},
    ],
)
def test_spec_resolver_guards(bad_override: dict) -> None:
    cat = Catalog.discover()
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "benchmarks": [{"type": "http_smoke"}],
    }
    base.update(bad_override)
    with pytest.raises(SpecResolutionError):
        resolve_spec(RunSpec.model_validate(base), cat)


def test_suite_override_and_skip_precedence() -> None:
    cat = Catalog.discover()
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "benchmark_suite": "smoke",
        "eval_suite": "smoke",
    }
    # Explicit benchmarks replaces benchmark_suite; evals=[] skips eval_suite.
    spec = RunSpec.model_validate(
        {**base, "benchmarks": [{"type": "http_smoke"}], "evals": []}
    )
    job = resolve_spec(spec, cat)[0]
    assert [w.definition.name for w in job.workloads] == ["http_smoke"]


def test_bundle_safety_and_deduplication(tmp_path: Path) -> None:
    w = DeduplicatedEvalRecordWriter(workload="ev", output_dir=tmp_path / "ev")
    for rep in (0, 1):
        w.write_response(
            dataset_id="ds",
            dataset_revision="r1",
            sample_id="s-0",
            question="2+2?",
            reference_answer="4",
            rendered_input="Q: 2+2?",
            point_id="c1",
            repeat=rep,
            concurrency=1,
            model_response="4",
            extracted_answer="4",
            score=1.0,
            correct=True,
        )
    paths = w.close()
    assert len(paths[0].read_text().splitlines()) == 1  # samples deduplicated
    assert len(paths[1].read_text().splitlines()) == 1  # inputs deduplicated
    assert len(paths[2].read_text().splitlines()) == 2  # both repeats kept

    # Path traversal tar entries are rejected.
    bad_tar = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(BundleValidationError):
        validate_bundle(bad_tar)


def test_all_three_plugins_and_end_to_end_gpu_smoke_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    assert_plugin_contract(
        workload_name="http_smoke",
        tmp_path=tmp_path / "http",
        overrides={"concurrency": [1, 2], "num_requests": 4},
    )
    res_dir, _ = assert_plugin_contract(
        workload_name="direct_smoke", tmp_path=tmp_path / "direct"
    )
    assert res_dir.measurements[0].metrics["hf_token_leaked"] == 0

    res_gsm, _ = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "gsm",
        overrides=_gsm8k_overrides(),
    )
    assert res_gsm.measurements[0].metrics["accuracy"] == 1.0

    # Full combined smoke_gpu_spec.yaml (with offline fixture overrides) + resealing.
    cat = Catalog.discover()
    raw = yaml.safe_load((FIXTURES / "smoke_gpu_spec.yaml").read_text())
    raw["evals"][0].update(_gsm8k_overrides())
    local_spec = tmp_path / "local_smoke.yaml"
    local_spec.write_text(yaml.safe_dump(raw), encoding="utf-8")

    job = resolve_spec(load_spec_file(local_spec), cat, attempt_id="att-e2e")[0]
    res, tar1 = Coordinator(launcher=FakeLauncher(), vllm_commit=VALID_SHA).run(
        job, tmp_path / "e2e", source_spec_path=local_spec
    )
    assert res.execution_status == ExecutionStatus.COMPLETED
    h1 = hashlib.sha256(tar1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(
        seal_bundle(tar1.parent, res.job_id, res.attempt_id).read_bytes()
    ).hexdigest()
    assert h1 == h2
    validate_bundle(tar1)

    # Replayable spec.resolved.yaml parses cleanly.
    assert load_spec_file(tar1.parent / "spec.resolved.yaml").schema_version == 1


def test_coordinator_no_silent_fallback_and_cli_smoke(tmp_path: Path) -> None:
    cat = Catalog.discover()
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
            "benchmarks": [
                {"type": "http_smoke", "concurrency": [1, 2], "num_requests": 2}
            ],
        }
    )
    job = resolve_spec(spec, cat, attempt_id="att-fb")[0]
    res, _ = Coordinator(launcher=FakeLauncher(allow_reset_cache=False)).run(
        job, tmp_path / "fb"
    )
    assert res.execution_status == ExecutionStatus.FAILED

    # CLI validate + fresh-interpreter `python -m tools.byog --help` (no cycles).
    assert (
        main(["validate", "--spec", str(FIXTURES / "smoke_gpu_spec.yaml")]) == EXIT_OK
    )
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "tools.byog", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: byog" in proc.stdout
