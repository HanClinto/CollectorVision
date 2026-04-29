#!/usr/bin/env python3
"""Benchmark CollectorVision ONNX models on CPU-class edge devices.

This is intended for Raspberry Pi and local CPU baselines.  Browser-specific
benchmarks live in ``examples/web_scanner/model_benchmark.html``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "cornelius": ROOT / "collector_vision/weights/cornelius.onnx",
    "milo": ROOT / "collector_vision/weights/milo.onnx",
}
DEFAULT_GITHUB_THREAD_URL = os.environ.get(
    "COLLECTORVISION_BENCHMARK_THREAD_URL",
    "https://github.com/HanClinto/CollectorVision/issues/16",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def run_text_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def run_git_command(args: list[str]) -> str | None:
    return run_text_command(["git", "-C", str(ROOT), *args])


def collect_source_info() -> dict[str, Any]:
    status = run_git_command(["status", "--porcelain"])
    return {
        "git_commit": run_git_command(["rev-parse", "HEAD"]),
        "git_commit_short": run_git_command(["rev-parse", "--short", "HEAD"]),
        "git_branch": run_git_command(["branch", "--show-current"]),
        "git_commit_date": run_git_command(["show", "-s", "--format=%cI", "HEAD"]),
        "git_remote_url": run_git_command(["config", "--get", "remote.origin.url"]),
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
    }


def model_metadata_from_session(session: ort.InferenceSession, model_path: Path) -> dict[str, Any]:
    meta = session.get_modelmeta()
    custom = dict(meta.custom_metadata_map)
    return {
        "version": custom.get("version") or custom.get("model_version"),
        "codename": custom.get("codename"),
        "task": custom.get("task"),
        "architecture": custom.get("architecture"),
        "base_model": custom.get("base_model"),
        "training_epochs": custom.get("training_epochs"),
        "input_size": custom.get("input_size"),
        "sha256": sha256_file(model_path),
        "onnx_producer_name": meta.producer_name,
        "onnx_graph_name": meta.graph_name,
        "onnx_domain": meta.domain,
        "onnx_model_version": meta.version,
        "custom_metadata": custom,
    }


def get_cpu_model() -> str:
    if sys.platform == "darwin":
        model = run_text_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
        chip = run_text_command(["sysctl", "-n", "machdep.cpu.brand"])
        if chip:
            return chip
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
    return platform.processor() or platform.machine() or "unknown"


def get_physical_cores() -> int | None:
    if sys.platform == "darwin":
        value = run_text_command(["sysctl", "-n", "hw.physicalcpu"])
        return int(value) if value and value.isdigit() else None
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            physical_ids: set[tuple[str, str]] = set()
            current_physical = "0"
            current_core: str | None = None
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    if current_core is not None:
                        physical_ids.add((current_physical, current_core))
                    current_physical = "0"
                    current_core = None
                    continue
                if line.startswith("physical id") and ":" in line:
                    current_physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id") and ":" in line:
                    current_core = line.split(":", 1)[1].strip()
            if current_core is not None:
                physical_ids.add((current_physical, current_core))
            if physical_ids:
                return len(physical_ids)
    return None


def get_total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = run_text_command(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE)
            if match:
                return int(match.group(1)) * 1024
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(pages)


def get_available_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        vm_stat = run_text_command(["vm_stat"])
        if vm_stat:
            page_size_match = re.search(r"page size of (\d+) bytes", vm_stat)
            page_size = int(page_size_match.group(1)) if page_size_match else 4096
            available_pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative"):
                match = re.search(rf"^{key}:\s+([\d.]+)", vm_stat, re.MULTILINE)
                if match:
                    available_pages += int(match.group(1).replace(".", ""))
            if available_pages:
                return available_pages * page_size
    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            text = meminfo.read_text(encoding="utf-8", errors="ignore")
            for key in ("MemAvailable", "MemFree"):
                match = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.MULTILINE)
                if match:
                    return int(match.group(1)) * 1024
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(pages)


def bytes_to_mib(value: int | None) -> float | None:
    return round(value / 1024 / 1024, 1) if value is not None else None


def collect_system_info() -> dict[str, Any]:
    total_memory = get_total_memory_bytes()
    available_memory = get_available_memory_bytes()
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": get_cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": get_physical_cores(),
        "memory_total_mib": bytes_to_mib(total_memory),
        "memory_available_mib": bytes_to_mib(available_memory),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def tensor_shape(session: ort.InferenceSession) -> list[int]:
    shape = session.get_inputs()[0].shape
    return [dim if isinstance(dim, int) else 1 for dim in shape]


def make_input(shape: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Use ImageNet-normalised-ish values instead of all-zero input so the graph
    # exercises the same operator paths as real images.
    return rng.normal(0.0, 1.0, size=shape).astype(np.float32)


def summarize_profile(profile_path: str | None) -> dict[str, Any] | None:
    if not profile_path:
        return None
    profile_file = Path(profile_path)
    if not profile_file.exists():
        return None

    with profile_file.open("r", encoding="utf-8") as fh:
        events = json.load(fh)

    by_op: dict[str, dict[str, float | int]] = {}
    by_node: dict[str, dict[str, float | int | str]] = {}
    for event in events:
        if event.get("cat") != "Node" or event.get("ph") != "X":
            continue
        args = event.get("args", {})
        op_name = args.get("op_name") or event.get("name", "unknown").split("_kernel_time")[0]
        node_name = event.get("name", "unknown").replace("_kernel_time", "")
        duration_ms = float(event.get("dur", 0.0)) / 1000.0
        entry = by_op.setdefault(op_name, {"count": 0, "total_ms": 0.0})
        entry["count"] = int(entry["count"]) + 1
        entry["total_ms"] = float(entry["total_ms"]) + duration_ms
        node_entry = by_node.setdefault(node_name, {"op": op_name, "count": 0, "total_ms": 0.0})
        node_entry["count"] = int(node_entry["count"]) + 1
        node_entry["total_ms"] = float(node_entry["total_ms"]) + duration_ms

    ranked_ops = sorted(by_op.items(), key=lambda item: float(item[1]["total_ms"]), reverse=True)
    ranked_nodes = sorted(by_node.items(), key=lambda item: float(item[1]["total_ms"]), reverse=True)
    return {
        "profile_path": str(profile_file.relative_to(ROOT) if profile_file.is_relative_to(ROOT) else profile_file),
        "note": "Profile includes first run, warmup runs, and measured runs; use it to locate hotspots, not exact steady-state latency.",
        "top_ops": [
            {
                "op": op,
                "count": stats["count"],
                "total_ms": round(float(stats["total_ms"]), 4),
                "avg_ms": round(float(stats["total_ms"]) / int(stats["count"]), 4),
            }
            for op, stats in ranked_ops[:12]
        ],
        "top_nodes": [
            {
                "node": node,
                "op": stats["op"],
                "count": stats["count"],
                "total_ms": round(float(stats["total_ms"]), 4),
                "avg_ms": round(float(stats["total_ms"]) / int(stats["count"]), 4),
            }
            for node, stats in ranked_nodes[:20]
        ],
    }


def benchmark_model(
    name: str,
    model_path: Path,
    threads: int,
    runs: int,
    warmup: int,
    provider: str,
    profile: bool,
    seed: int,
    ort_log_severity: int,
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.log_severity_level = ort_log_severity
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        profile_dir = ROOT / "build/model_benchmarks/profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        provider_label = provider.replace("ExecutionProvider", "").lower()
        options.profile_file_prefix = str(profile_dir / f"{name}-{provider_label}-threads{threads}")

    create_start = time.perf_counter()
    providers = provider_chain(provider)
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    session_create_ms = (time.perf_counter() - create_start) * 1000.0
    model_metadata = model_metadata_from_session(session, model_path)

    input_meta = session.get_inputs()[0]
    shape = tensor_shape(session)
    feeds = {input_meta.name: make_input(shape, seed)}

    first_start = time.perf_counter()
    session.run(None, feeds)
    first_run_ms = (time.perf_counter() - first_start) * 1000.0

    # Exclude warmup runs from steady-state latency.  The first run above is
    # reported separately because ONNX Runtime and CPU caches often make it
    # slower than later runs, especially on Raspberry Pi class hardware.
    for _ in range(warmup):
        session.run(None, feeds)

    timings: list[float] = []
    for _ in range(runs):
        run_start = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - run_start) * 1000.0)

    profile_path = session.end_profiling() if profile else None

    return {
        "name": name,
        "path": str(model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path),
        "size_mib": round(model_path.stat().st_size / 1024 / 1024, 4),
        "model": model_metadata,
        "provider": provider,
        "provider_chain": providers,
        "session_providers": session.get_providers(),
        "threads": threads,
        "input_shape": shape,
        "outputs": [output.name for output in session.get_outputs()],
        "session_create_ms": round(session_create_ms, 4),
        "first_run_ms": round(first_run_ms, 4),
        "latency_ms": {
            "runs": runs,
            "warmup": warmup,
            "note": "first_run_ms and warmup runs are excluded from these steady-state timings",
            "mean": round(statistics.fmean(timings), 4),
            "median": round(statistics.median(timings), 4),
            "p90": round(percentile(timings, 90), 4),
            "p95": round(percentile(timings, 95), 4),
            "min": round(min(timings), 4),
            "max": round(max(timings), 4),
        },
        "profile": summarize_profile(profile_path),
    }


def parse_model_specs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return DEFAULT_MODELS
    models: dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            name = Path(raw_path).stem
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        models[name] = path
    return models


def parse_provider_specs(spec: str) -> list[str]:
    available = ort.get_available_providers()
    if spec in {"all", "all-local"}:
        return [provider for provider in available if not is_remote_provider(provider)]
    if spec in {"all-with-remote", "all-available"}:
        return available
    return [part.strip() for part in spec.split(",") if part.strip()]


def is_remote_provider(provider: str) -> bool:
    return provider in {"AzureExecutionProvider"}


def provider_chain(provider: str) -> list[str]:
    if provider == "CPUExecutionProvider":
        return [provider]
    if "CPUExecutionProvider" in ort.get_available_providers():
        return [provider, "CPUExecutionProvider"]
    return [provider]


def failed_benchmark_result(
    name: str,
    model_path: Path,
    threads: int,
    provider: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path),
        "size_mib": round(model_path.stat().st_size / 1024 / 1024, 4) if model_path.exists() else None,
        "provider": provider,
        "provider_chain": provider_chain(provider),
        "threads": threads,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def format_markdown_report(report: dict[str, Any], output: Path) -> str:
    system = report["system"]
    source = report["source"]
    commit = source.get("git_commit_short") or "unknown"
    dirty = "dirty" if source.get("git_dirty") else "clean"
    cpu = system.get("cpu_model") or system.get("processor") or system.get("machine") or "unknown CPU"
    cores = f"{system.get('cpu_physical_cores') or '?'}P/{system.get('cpu_logical_cores') or '?'}L"
    memory = f"{system.get('memory_total_mib') or '?'} MiB RAM"
    available = system.get("memory_available_mib")
    if available is not None:
        memory = f"{memory}, {available} MiB available"

    rows = []
    for result in report["benchmarks"]:
        if "error" in result:
            rows.append(
                "| {name} | — | {threads} | {provider} | error | — | — | — | {size} |".format(
                    name=result["name"],
                    threads=result["threads"],
                    provider=result["provider"],
                    size=result.get("size_mib") or "—",
                )
            )
        else:
            latency = result["latency_ms"]
            model = result.get("model", {})
            rows.append(
                "| {name} | {version} | {threads} | {provider} | {mean:.2f} | {median:.2f} | {p90:.2f} | {first:.2f} | {size:.2f} |".format(
                    name=result["name"],
                    version=model.get("version") or "unknown",
                    threads=result["threads"],
                    provider=result["provider"],
                    mean=latency["mean"],
                    median=latency["median"],
                    p90=latency["p90"],
                    first=result["first_run_ms"],
                    size=result["size_mib"],
                )
            )

    report_json = json.dumps(report, indent=2)
    output_label = str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    return "\n".join(
        [
            f"## ONNX CPU benchmark — {cpu}",
            "",
            f"- **Source:** `{commit}` ({dirty}) on `{source.get('git_branch') or 'unknown'}`",
            f"- **System:** `{system['platform']}` · `{cores}` · `{memory}`",
            f"- **Runtime:** ONNX Runtime `{system['onnxruntime']}` · Python `{system['python']}`",
            "",
            "Mean/median/P90 are steady-state ms; first run and warmups are excluded.",
            "",
            "| Model | Version | Threads | Provider | Mean ms | Median ms | P90 ms | First ms | Size MiB |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "<details>",
            f"<summary>Full JSON report ({output_label})</summary>",
            "",
            "```json",
            report_json,
            "```",
            "</details>",
            "",
        ]
    )


def parse_github_thread_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 4:
        return None
    owner, repo, kind, number = parts[:4]
    if kind not in {"issues", "discussions"} or not number.isdigit():
        return None
    return f"{owner}/{repo}", kind, number


def run_gh_comment(thread_url: str, body: str) -> bool:
    parsed = parse_github_thread_url(thread_url)
    gh = shutil.which("gh")
    if not parsed or not gh:
        return False

    repo, kind, number = parsed
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as fh:
        fh.write(body)
        body_path = fh.name
    try:
        if kind == "issues":
            command = [gh, "issue", "comment", number, "--repo", repo, "--body-file", body_path]
        else:
            command = [gh, "discussion", "comment", number, "--repo", repo, "--body-file", body_path]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Posted benchmark result to {thread_url}")
            return True
        print("gh could not post the benchmark result:", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return False
    finally:
        Path(body_path).unlink(missing_ok=True)


def copy_to_clipboard(text: str) -> bool:
    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["pbcopy"])
    elif sys.platform == "win32":
        commands.append(["clip"])
    else:
        commands.extend([["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]])

    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, input=text, text=True, check=False)
        if result.returncode == 0:
            return True
    return False


def open_browser_comment_fallback(thread_url: str, body: str, output: Path) -> None:
    comment_path = output.with_suffix(".github-comment.md")
    comment_path.write_text(body, encoding="utf-8")
    copied = copy_to_clipboard(body)
    print(f"Wrote GitHub comment body to {comment_path}")
    if copied:
        print("Copied GitHub comment body to the clipboard.")
    else:
        print("Could not copy to clipboard automatically; open the .github-comment.md file and copy it manually.")
    if thread_url:
        webbrowser.open(thread_url)
        print(f"Opened {thread_url}")


def wants_to_contribute(mode: str, thread_url: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    if not thread_url or not sys.stdin.isatty():
        return False
    answer = input("Contribute this benchmark result as a GitHub comment? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def contribute_report(report: dict[str, Any], output: Path, thread_url: str, mode: str, method: str) -> None:
    if not wants_to_contribute(mode, thread_url):
        if mode != "never" and not thread_url:
            print(
                "Benchmark contribution skipped. Set --github-thread-url or "
                "COLLECTORVISION_BENCHMARK_THREAD_URL once a static issue/discussion exists."
            )
        return

    body = format_markdown_report(report, output)
    if method in {"auto", "gh"}:
        if run_gh_comment(thread_url, body):
            return
        if method == "gh":
            print("Falling back skipped because --contribution-method=gh was requested.")
            return

    open_browser_comment_fallback(thread_url, body, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="Model spec: name=path or path. Defaults to Cornelius and Milo.")
    parser.add_argument("--threads", default="1,2,4", help="Comma-separated intra-op thread counts.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs per model/thread combination. Excluded from reported steady-state latency.")
    parser.add_argument(
        "--provider",
        default="all-local",
        help="ONNX Runtime provider, comma-separated providers, 'all'/'all-local' for local providers only, or 'all-with-remote'/'all-available' to include remote-capable providers such as AzureExecutionProvider. Non-CPU providers use CPU fallback for unsupported nodes.",
    )
    parser.add_argument("--ort-log-severity", type=int, default=3, help="ONNX Runtime log severity: 0 verbose, 1 info, 2 warning, 3 error, 4 fatal. Default suppresses provider warning noise during sweeps.")
    parser.add_argument("--profile", action="store_true", help="Write and summarize ONNX Runtime node profiles.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=None, help="JSON output path. Defaults under build/model_benchmarks/.")
    parser.add_argument(
        "--contribute",
        choices=("ask", "always", "never"),
        default="ask",
        help="Offer to contribute the result to the configured GitHub benchmark thread.",
    )
    parser.add_argument(
        "--github-thread-url",
        default=DEFAULT_GITHUB_THREAD_URL,
        help="Static GitHub issue/discussion URL for collecting benchmark comments. Can also be set with COLLECTORVISION_BENCHMARK_THREAD_URL.",
    )
    parser.add_argument(
        "--contribution-method",
        choices=("auto", "gh", "browser"),
        default="auto",
        help="How to contribute results: gh CLI, browser/clipboard fallback, or auto.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / f"build/model_benchmarks/onnx-cpu-{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    models = parse_model_specs(args.model)
    thread_counts = [int(part.strip()) for part in args.threads.split(",") if part.strip()]
    providers = parse_provider_specs(args.provider)

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": collect_source_info(),
        "system": collect_system_info(),
        "benchmarks": [],
    }

    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(path)
        for provider in providers:
            for threads in thread_counts:
                try:
                    result = benchmark_model(
                        name=name,
                        model_path=path,
                        threads=threads,
                        runs=args.runs,
                        warmup=args.warmup,
                        provider=provider,
                        profile=args.profile,
                        ort_log_severity=args.ort_log_severity,
                        seed=args.seed,
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should keep sweeping providers
                    result = failed_benchmark_result(name, path, threads, provider, exc)
                    print(f"{name:10s} provider={provider:<24s} threads={threads:<2d} ERROR {exc}")
                else:
                    latency = result["latency_ms"]
                    model = result["model"]
                    print(
                        f"{name:10s} provider={provider:<24s} threads={threads:<2d} "
                        f"mean={latency['mean']:8.3f} ms p90={latency['p90']:8.3f} ms "
                        f"first={result['first_run_ms']:8.3f} ms size={result['size_mib']:6.3f} MiB "
                        f"version={model.get('version') or 'unknown'} hash={(model.get('sha256') or 'unknown')[:19]}"
                    )
                report["benchmarks"].append(result)

    with output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    contribute_report(report, output, args.github_thread_url, args.contribute, args.contribution_method)


if __name__ == "__main__":
    main()
