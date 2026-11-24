# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example HTTP server-mode benchmark plugin."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

from tools.byog.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.execution import ExecutionStatus


def _pct(vals: list[float], p: float) -> float:
    return vals[min(len(vals) - 1, max(0, math.ceil(p * len(vals)) - 1))]


class HttpSmokeBenchmark(BenchmarkRunner):
    """Minimal HTTP benchmark exercising `/v1/completions` with excluded warmup."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concs = params.get("concurrency", [1])
        if isinstance(concs, int):
            concs = [concs]
        return [
            WorkloadPoint(f"c{int(c)}", {**params, "concurrency": int(c)})
            for c in concs
        ]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        if not isinstance(ctx, ServerRunContext):
            raise TypeError("HttpSmokeBenchmark requires ServerRunContext.")
        p = point.params
        conc = int(p.get("concurrency", 1))
        n_req = int(p.get("num_requests", 8))
        n_warm = int(p.get("warmup_requests", 2))
        max_toks = int(p.get("max_tokens", 32))
        prompt = str(p.get("prompt", "Write a short sentence about vLLM:"))
        deadline = time.monotonic() + max(1, ctx.timeout_s)

        def _one(_: int) -> tuple[float | None, int, str | None, str | None]:
            body = json.dumps(
                {
                    "model": ctx.model,
                    "prompt": prompt,
                    "max_tokens": max_toks,
                    "temperature": 0.0,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{ctx.base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.monotonic()
            try:
                timeout = min(60.0, max(1.0, deadline - time.monotonic()))
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                toks = int((parsed.get("usage") or {}).get("completion_tokens") or 0)
                return time.monotonic() - t0, toks, None, None
            except urllib.error.HTTPError as err:
                kind = "model" if 400 <= err.code < 500 else "infra"
                return None, 0, kind, f"HTTP {err.code}: {err.reason}"
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ) as err:
                return None, 0, "infra", str(err)
            except (ValueError, KeyError, TypeError) as err:
                return None, 0, "model", str(err)

        for i in range(max(0, n_warm)):
            _one(i)

        t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, conc)) as pool:
            outcomes = list(pool.map(_one, range(n_req)))
        elapsed = max(1e-6, time.monotonic() - t_start)

        lats = sorted(lat for lat, _, _, _ in outcomes if lat is not None)
        toks = sum(t for lat, t, _, _ in outcomes if lat is not None)
        m_err = sum(1 for _, _, k, _ in outcomes if k == "model")
        i_err = sum(1 for _, _, k, _ in outcomes if k == "infra")
        first_err = next((e for _, _, _, e in outcomes if e), None)
        n = len(lats)

        metrics: dict[str, float | int | str | None] = {
            "throughput_req_s": n / elapsed,
            "output_tokens_per_s": toks / elapsed,
            "mean_latency_s": (sum(lats) / n) if n else None,
            "p50_latency_s": _pct(lats, 0.50) if n else None,
        }
        if n >= 100:
            metrics["p90_latency_s"] = _pct(lats, 0.90)
        if n >= 1000:
            metrics["p99_latency_s"] = _pct(lats, 0.99)

        return PointResult(
            status=ExecutionStatus.COMPLETED if n > 0 else ExecutionStatus.FAILED,
            metrics=metrics,
            attempted=n_req,
            completed=n,
            model_errors=m_err,
            infra_errors=i_err,
            error_message=first_err if n == 0 else None,
        )
