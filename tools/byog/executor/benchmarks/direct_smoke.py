# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example direct-mode (non-server) benchmark plugin."""

from __future__ import annotations

import json
import sys
from typing import Any

from tools.byog.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.execution import ExecutionStatus

_PROBE = (
    "import json, os, time; t0 = time.monotonic(); "
    "s = sum(i * i for i in range(2000)); "
    "print(json.dumps({'elapsed_s': max(1e-6, time.monotonic() - t0), "
    "'checksum': s, 'hf_token_leaked': 'HF_TOKEN' in os.environ}))"
)


class DirectSmokeBenchmark(BenchmarkRunner):
    """Minimal direct-mode benchmark running via `ctx.run_command`."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        cases = params.get("cases")
        if isinstance(cases, list) and cases:
            return [WorkloadPoint(str(c), {**params, "case": str(c)}) for c in cases]
        return [WorkloadPoint("default", dict(params))]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        if not isinstance(ctx, DirectRunContext):
            raise TypeError("DirectSmokeBenchmark requires DirectRunContext.")
        iters = int(point.params.get("iterations", 2))
        res = ctx.run_command([sys.executable, "-c", _PROBE], ctx.timeout_s)
        if res.timed_out:
            return PointResult(
                status=ExecutionStatus.TIMED_OUT,
                attempted=iters,
                completed=0,
                infra_errors=1,
                error_message="Direct smoke subprocess timed out.",
            )
        if res.returncode != 0:
            return PointResult(
                status=ExecutionStatus.FAILED,
                attempted=iters,
                completed=0,
                infra_errors=1,
                error_message=f"Direct smoke exited {res.returncode}: {res.stderr}",
            )
        try:
            parsed = json.loads(res.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            return PointResult(
                status=ExecutionStatus.FAILED,
                attempted=iters,
                completed=0,
                infra_errors=1,
                error_message=f"Invalid JSON from direct smoke: {exc}",
            )
        return PointResult(
            status=ExecutionStatus.COMPLETED,
            metrics={
                "elapsed_s": float(parsed["elapsed_s"]),
                "wall_s": res.elapsed_s,
                "hf_token_leaked": int(bool(parsed.get("hf_token_leaked"))),
            },
            attempted=iters,
            completed=iters,
        )
