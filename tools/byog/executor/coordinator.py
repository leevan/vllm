# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor / Coordinator (C4): owns server lifecycle, cleanup, and result bundling."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.byog.bundle import (
    DeduplicatedEvalRecordWriter,
    seal_bundle,
    write_bundle_metadata,
)
from tools.byog.catalog import instantiate_plugin
from tools.byog.interfaces import (
    BenchmarkRunner,
    CacheResetUnsupportedError,
    DirectCommandResult,
    DirectRunContext,
    EvalRunner,
    PlatformAdapter,
    PointResult,
    ServerHandle,
    ServerRunContext,
    ServerTransportError,
    WorkloadPoint,
)
from tools.byog.schemas.execution import (
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    ResolvedJob,
    ResolvedWorkload,
    WorkloadKind,
)
from tools.byog.schemas.results import MeasurementRecord, RunResult

HARNESS_VERSION = "0.1.0-phase0"


class _JobCancelled(BaseException):
    pass


class Coordinator:
    """Runs one `ResolvedJob`, owns server lifecycle, and seals `bundle.tar.gz`."""

    def __init__(
        self,
        *,
        launcher: PlatformAdapter,
        image_ref: str | None = None,
        vllm_commit: str | None = None,
        image_digest: str | None = None,
    ):
        self._launcher = launcher
        self._image_ref = image_ref
        self._vllm_commit = vllm_commit
        self._image_digest = image_digest

    def run(
        self,
        job: ResolvedJob,
        output_root: Path,
        *,
        source_spec_path: Path | None = None,
    ) -> tuple[RunResult, Path]:
        attempt_id = job.attempt_id or f"att-{uuid.uuid4().hex[:16]}"
        run_job = job.model_copy(update={"attempt_id": attempt_id})
        bundle_dir = output_root / run_job.job_id / attempt_id
        logs_dir = bundle_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        deadline = t0 + run_job.timeout_s
        hw = self._launcher.detect_hardware()
        base_mem_list = (
            list(hw.device_memory_used_mib)
            if hw.device_memory_used_mib is not None
            else None
        )

        measurements: list[MeasurementRecord] = []
        errors: list[str] = []
        cleanups: list[dict[str, Any]] = []
        status = ExecutionStatus.COMPLETED
        active: ServerHandle | None = None
        active_dev = False

        def _rem_s() -> int:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"Job exceeded timeout_s={job.timeout_s}s.")
            return max(1, int(rem))

        def _stop_server() -> bool:
            nonlocal active, active_dev
            if active is None:
                return True
            h, active, active_dev = active, None, False
            rep = self._launcher.stop(h, baseline_memory_mib=base_mem_list)
            cleanups.append(asdict(rep))
            ok = True
            if not rep.terminated_pgid:
                errors.append("Server process group failed to terminate.")
                ok = False
            if rep.vram_reclaimed is False:
                errors.append("GPU VRAM was not reclaimed to baseline.")
                ok = False
            return ok

        def _ensure_server(dev_mode: bool) -> ServerHandle:
            nonlocal active, active_dev
            if active is not None and active_dev == dev_mode:
                return active
            if active is not None and not _stop_server():
                raise RuntimeError("Aborting launch after failed server cleanup.")
            active = self._launcher.start(
                job.serve_config,
                job.serve_env,
                logs_dir,
                _rem_s(),
                enable_dev_mode=dev_mode,
            )
            active_dev = dev_mode
            return active

        def _record(
            w: ResolvedWorkload,
            pt: WorkloadPoint,
            actual_lc: LifecyclePolicy,
            pr: PointResult,
            *,
            fallback_reason: str | None = None,
        ) -> None:
            defn = w.definition
            rels = []
            root = bundle_dir.resolve()
            for p in pr.artifacts:
                try:
                    rels.append(p.resolve().relative_to(root).as_posix())
                except ValueError:
                    rels.append(p.name)
            measurements.append(
                MeasurementRecord(
                    workload=defn.name,
                    kind=defn.kind,
                    point_id=pt.point_id,
                    execution_mode=defn.execution_mode,
                    protocol_version=defn.protocol_version,
                    requested_lifecycle=defn.lifecycle,
                    actual_lifecycle=actual_lc,
                    lifecycle_fallback_reason=fallback_reason,
                    status=pr.status,
                    metrics=pr.metrics,
                    attempted=pr.attempted,
                    completed=pr.completed,
                    scoring_denominator=pr.scoring_denominator,
                    model_errors=pr.model_errors,
                    infra_errors=pr.infra_errors,
                    error_message=pr.error_message,
                    artifacts=rels,
                )
            )

        prev_sig = None
        if threading.current_thread() is threading.main_thread():
            prev_sig = signal.getsignal(signal.SIGTERM)
            with contextlib.suppress(ValueError, OSError):
                signal.signal(
                    signal.SIGTERM,
                    lambda s, _f: (_ for _ in ()).throw(_JobCancelled(f"Signal {s}")),
                )

        try:
            if (
                job.hardware.gpu
                and hw.gpu_id != "unknown"
                and hw.gpu_id not in job.hardware.gpu
            ):
                raise RuntimeError(
                    f"Observed GPU '{hw.gpu_id}' not in {job.hardware.gpu}."
                )

            served_model = str(
                job.serve_config.get("served-model-name")
                or job.serve_config.get("served_model_name")
                or job.serve_config.get("model", "")
            )

            for w in job.workloads:
                defn = w.definition
                plugin = instantiate_plugin(defn.entrypoint, defn.kind)
                pts = plugin.points(w.params)
                w_out = (
                    bundle_dir
                    / ("evals" if defn.kind == WorkloadKind.EVAL else "benchmarks")
                    / defn.name
                )
                w_out.mkdir(parents=True, exist_ok=True)

                if defn.execution_mode == ExecutionMode.DIRECT:
                    if not _stop_server():
                        break
                    assert isinstance(plugin, BenchmarkRunner)
                    for pt in pts:
                        rem = _rem_s()

                        def _run_cmd(
                            cmd: list[str], t: int | None = None
                        ) -> DirectCommandResult:
                            b = _rem_s()
                            return self._launcher.run_direct(
                                cmd, job.serve_env, min(b, t) if t else b
                            )

                        ctx = DirectRunContext(
                            serve_config=dict(job.serve_config),
                            serve_env=dict(job.serve_env),
                            output_dir=w_out,
                            timeout_s=rem,
                            run_command=_run_cmd,
                        )
                        try:
                            pr = plugin.run_point(pt, ctx)
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        _record(w, pt, LifecyclePolicy.NONE, pr)
                    continue

                req_lc = defn.lifecycle
                dev_mode = req_lc == LifecyclePolicy.RESET_CACHE
                writer = (
                    DeduplicatedEvalRecordWriter(workload=defn.name, output_dir=w_out)
                    if defn.kind == WorkloadKind.EVAL
                    else None
                )
                try:
                    for idx, pt in enumerate(pts):
                        actual_lc = req_lc
                        fb_reason: str | None = None
                        if idx == 0:
                            handle = _ensure_server(dev_mode)
                        elif req_lc == LifecyclePolicy.RESTART_SERVER:
                            if not _stop_server():
                                break
                            handle = _ensure_server(dev_mode)
                        elif req_lc == LifecyclePolicy.RESET_CACHE:
                            handle = _ensure_server(dev_mode)
                            try:
                                self._launcher.reset_cache(handle)
                                actual_lc = LifecyclePolicy.RESET_CACHE
                            except CacheResetUnsupportedError as exc:
                                if not defn.allow_lifecycle_fallback:
                                    msg = f"reset_cache unsupported: {exc}"
                                    errors.append(msg)
                                    _record(
                                        w,
                                        pt,
                                        actual_lc,
                                        PointResult(
                                            status=ExecutionStatus.FAILED,
                                            error_message=msg,
                                        ),
                                    )
                                    break
                                fb_reason = str(exc)
                                if not _stop_server():
                                    break
                                handle = _ensure_server(False)
                                actual_lc = LifecyclePolicy.RESTART_SERVER
                            except ServerTransportError as exc:
                                msg = f"reset_cache transport error: {exc}"
                                errors.append(msg)
                                _record(
                                    w,
                                    pt,
                                    actual_lc,
                                    PointResult(
                                        status=ExecutionStatus.FAILED,
                                        error_message=msg,
                                    ),
                                )
                                break
                        else:
                            handle = _ensure_server(dev_mode)

                        s_ctx = ServerRunContext(
                            base_url=handle.base_url,
                            model=served_model,
                            output_dir=w_out,
                            timeout_s=_rem_s(),
                        )
                        try:
                            if defn.kind == WorkloadKind.EVAL:
                                assert isinstance(plugin, EvalRunner)
                                assert writer is not None
                                pr = plugin.run_point(pt, s_ctx, writer)
                            else:
                                assert isinstance(plugin, BenchmarkRunner)
                                pr = plugin.run_point(pt, s_ctx)
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        _record(w, pt, actual_lc, pr, fallback_reason=fb_reason)
                finally:
                    if writer is not None:
                        rec_paths = writer.close()
                        root = bundle_dir.resolve()
                        rels = [
                            p.resolve().relative_to(root).as_posix() for p in rec_paths
                        ]
                        for m in measurements:
                            if m.workload == defn.name:
                                m.artifacts.extend(
                                    r for r in rels if r not in m.artifacts
                                )

            if errors or any(
                m.status != ExecutionStatus.COMPLETED for m in measurements
            ):
                status = ExecutionStatus.FAILED
        except _JobCancelled as exc:
            errors.append(str(exc))
            status = ExecutionStatus.CANCELLED
        except TimeoutError as exc:
            errors.append(str(exc))
            status = ExecutionStatus.TIMED_OUT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            status = ExecutionStatus.FAILED
        finally:
            if not _stop_server() and status == ExecutionStatus.COMPLETED:
                status = ExecutionStatus.FAILED
            if prev_sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(signal.SIGTERM, prev_sig)

        if cleanups:
            (logs_dir / "cleanup.json").write_text(
                json.dumps(cleanups, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        result = RunResult(
            job_id=run_job.job_id,
            attempt_id=attempt_id,
            spec_name=run_job.spec_name,
            execution_status=status,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_s=max(0.0, time.monotonic() - t0),
            harness_version=HARNESS_VERSION,
            vllm_commit=self._vllm_commit,
            image_digest=self._image_digest,
            requested_hardware=run_job.hardware.model_dump(),
            observed_hardware=asdict(hw),
            resolved_config_hash=run_job.resolved_config_hash,
            measurements=measurements,
            errors=errors,
        )
        write_bundle_metadata(
            bundle_dir,
            job=run_job,
            result=result,
            source_spec_path=source_spec_path,
            image_ref=self._image_ref,
        )
        return result, seal_bundle(bundle_dir, run_job.job_id, attempt_id)
