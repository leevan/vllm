# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FakeLauncher and reusable `assert_plugin_contract()` helper for CPU tests."""

from __future__ import annotations

import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.coordinator import Coordinator
from tools.byog.interfaces import (
    CacheResetUnsupportedError,
    CleanupResult,
    DetectedHardware,
    DirectCommandResult,
    PlatformAdapter,
    ServerHandle,
    ServerTransportError,
)
from tools.byog.schemas.execution import (
    ExecutionMode,
    ExecutionStatus,
    RunSpec,
    WorkloadKind,
)
from tools.byog.schemas.results import RunResult
from tools.byog.spec import resolve_spec

_DIRECT_ENV = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH")
_TEST_SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeLauncher(PlatformAdapter):
    """In-process PlatformAdapter backed by `FakeVLLMServer`."""

    def __init__(
        self,
        *,
        allow_reset_cache: bool = True,
        fail_reset_transport: bool = False,
        fail_on_calls: set[int] | None = None,
        wrong_answer_mode: bool = False,
        vram_reclaimed: bool | None = True,
    ):
        self.allow_reset_cache = allow_reset_cache
        self.fail_reset_transport = fail_reset_transport
        self.fail_on_calls = set(fail_on_calls or set())
        self.wrong_answer_mode = wrong_answer_mode
        self.vram_reclaimed = vram_reclaimed
        self.starts = self.stops = self.resets = self.direct_calls = 0
        self.states: list[FakeServerState] = []
        self._active: dict[str, FakeVLLMServer] = {}

    @property
    def total_requests(self) -> int:
        return sum(s.calls for s in self.states)

    def detect_hardware(self) -> DetectedHardware:
        return DetectedHardware(
            platform="nvidia",
            gpu_id="h200",
            gpu_count=1,
            gpu_name="NVIDIA H200 (fake)",
            driver_version="570.00-fake",
            device_memory_used_mib=[0],
        )

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        self.starts += 1
        model = str(serve_config.get("model", "fake-model"))
        st = FakeServerState(
            model_name=model,
            allow_reset_cache=self.allow_reset_cache and enable_dev_mode,
            fail_on_calls=set(self.fail_on_calls),
            wrong_answer_mode=self.wrong_answer_mode,
        )
        self.states.append(st)
        srv = FakeVLLMServer(st)
        url = srv.start()
        self._active[url] = srv
        log_path = log_dir / f"fake_{self.starts}.log"
        log_path.write_text("fake server\n", encoding="utf-8")
        return ServerHandle(base_url=url, command=["fake"], log_path=log_path)

    def reset_cache(self, handle: ServerHandle) -> None:
        self.resets += 1
        if self.fail_reset_transport:
            raise ServerTransportError("simulated transport error")
        srv = self._active.get(handle.base_url)
        if srv is None or not srv.state.allow_reset_cache:
            raise CacheResetUnsupportedError("HTTP 404")
        srv.state.reset_calls += 1

    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> CleanupResult:
        self.stops += 1
        if srv := self._active.pop(handle.base_url, None):
            srv.stop()
        return CleanupResult(
            terminated_pgid=True,
            vram_reclaimed=self.vram_reclaimed,
            baseline_memory_mib=baseline_memory_mib,
        )

    def run_direct(
        self, cmd: list[str], serve_env: dict[str, str], timeout_s: int
    ) -> DirectCommandResult:
        self.direct_calls += 1
        env = {k: v for k, v in os.environ.items() if k in _DIRECT_ENV}
        env.update(serve_env)
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return DirectCommandResult(
            proc.returncode, proc.stdout, proc.stderr, time.monotonic() - t0
        )


def assert_plugin_contract(
    *,
    workload_name: str,
    tmp_path: Path,
    catalog: Catalog | None = None,
    launcher: FakeLauncher | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[RunResult, Path]:
    """Run one plugin against `FakeLauncher` and verify end-to-end contract."""
    cat = catalog or Catalog.discover()
    defn = cat.get_workload(workload_name)
    key = "benchmarks" if defn.kind == WorkloadKind.BENCHMARK else "evals"
    id_field = "type" if defn.kind == WorkloadKind.BENCHMARK else "name"
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "fake/m", "revision": _TEST_SHA}},
            "hardware": {"gpu": ["h200"]},
            key: [{id_field: workload_name, **(overrides or {})}],
        }
    )
    job = resolve_spec(spec, cat, attempt_id=f"att-{workload_name}")[0]
    active_launcher = launcher or FakeLauncher()
    res, archive = Coordinator(launcher=active_launcher).run(job, tmp_path / "out")

    assert res.execution_status == ExecutionStatus.COMPLETED, res.errors
    assert res.measurements and active_launcher.stops == active_launcher.starts
    for m in res.measurements:
        assert m.metrics and (m.attempted or 0) >= 1 and (m.completed or 0) >= 1
    if defn.execution_mode == ExecutionMode.SERVER:
        assert active_launcher.total_requests >= 1
    else:
        assert active_launcher.direct_calls >= 1

    validate_bundle(archive)
    if defn.kind == WorkloadKind.EVAL:
        with tarfile.open(archive, "r:gz") as tar:
            fh = tar.extractfile(f"evals/{workload_name}/responses.jsonl")
            assert fh is not None
            resp_txt = fh.read().decode()
            assert len(resp_txt.splitlines()) == sum(
                m.attempted or 0 for m in res.measurements
            )
    return res, archive
