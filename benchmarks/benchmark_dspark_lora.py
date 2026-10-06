#!/usr/bin/env python3
"""Streaming vLLM benchmark with exact token-ID prompts and output counts."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import secrets
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re

PRESETS = {"smoke": (512, 128, 8), "full": (8192, 4096, 8), "profile": (8192, 128, 8)}
METRIC_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)")
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def http(base, path, payload=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base.rstrip("/") + path, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return response.read().decode()


def metrics(text):
    result = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        labels = tuple(sorted((key, json.loads('"' + val + '"')) for key, val in LABEL_RE.findall(match[2] or "")))
        result[(match[1], labels)] = value
    return result


def metric_delta(before, after):
    changes = {}
    reset = []
    for key, value in after.items():
        name, labels = key
        # Counters only. Histograms and gauges remain available in raw snapshots.
        if not name.endswith("_total"):
            continue
        delta = value - before.get(key, 0.0)
        if delta < 0:
            reset.append({"name": name, "labels": dict(labels)})
        else:
            changes[name] = changes.get(name, 0.0) + delta
    drafts = changes.get("vllm:spec_decode_num_drafts_total")
    proposed = changes.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = changes.get("vllm:spec_decode_num_accepted_tokens_total")
    return {
        "counter_deltas": changes,
        "counter_resets": reset,
        "drafts": drafts,
        "draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "draft_acceptance_rate": accepted / proposed if accepted is not None and proposed and not reset else None,
        "mean_acceptance_length_including_bonus": 1 + accepted / drafts
        if accepted is not None and drafts and not reset
        else None,
        "scope": (
            "Global server counters across snapshot interval; "
            "valid request attribution requires an otherwise idle server."
        ),
    }


def percentile(values, percent):
    if not values:
        return None
    seq = sorted(values)
    pos = (len(seq) - 1) * percent / 100
    low, high = math.floor(pos), math.ceil(pos)
    return seq[low] + (seq[high] - seq[low]) * (pos - low)


def distribution(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def generate_prompt(seed, index, count, vocab_size, banned):
    rng = random.Random(f"{seed}:prompt:{index}")
    result = []
    while len(result) < count:
        token = rng.randrange(2, vocab_size)
        if token not in banned:
            result.append(token)
    return result


def request_once(args, index, prompt, gate=None):
    """Validate the streamed token IDs, finish reason, and usage for one request."""
    if gate is not None:
        gate.wait()
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "n": 1,
        "echo": False,
    }
    start = time.perf_counter()
    result = {
        "index": index,
        "requested_prompt_tokens": len(prompt),
        "requested_output_tokens": args.output_tokens,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
        "start_monotonic_s": start,
        "status": "error",
    }
    first = last = None
    token_ids, bursts, text_parts = [], [], []
    prompt_ids, usage, finish, done = None, None, None, False
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": f"bench-{args.seed}-{index}"},
    )
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=args.timeout
        ) as response:
            data_lines = []
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line or not data_lines:
                    continue
                data = "\n".join(data_lines)
                data_lines = []
                now = time.perf_counter()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(json.dumps(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise RuntimeError("Unexpected second completion choice")
                    if choice.get("prompt_token_ids") is not None:
                        prompt_ids = choice["prompt_token_ids"]
                    delta = choice.get("token_ids")
                    if delta is None:
                        delta = []
                    if delta:
                        first = now if first is None else first
                        last = now
                        token_ids.extend(delta)
                        bursts.append({"elapsed_s": now - start, "tokens": len(delta)})
                    if choice.get("text"):
                        text_parts.append(choice["text"])
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        end = time.perf_counter()
        errors = []
        if not done:
            errors.append("missing SSE DONE marker")
        if first is None:
            errors.append("no returned output token IDs; TTFT and exact output counts cannot be established")
        if len(token_ids) != args.output_tokens:
            errors.append(f"returned token IDs {len(token_ids)} != expected {args.output_tokens}")
        if usage is None:
            errors.append("missing final usage")
        else:
            if usage.get("prompt_tokens") != len(prompt):
                errors.append(f"usage prompt_tokens {usage.get('prompt_tokens')} != {len(prompt)}")
            if usage.get("completion_tokens") != len(token_ids):
                errors.append("usage completion_tokens differs from returned token IDs")
        if prompt_ids is not None and prompt_ids != prompt:
            errors.append("returned prompt token IDs differ from exact submitted IDs")
        if finish != "length":
            errors.append(f"finish_reason {finish!r}, expected length with ignore_eos")
        result.update(
            status="success" if not errors else "invalid",
            errors=errors,
            finish_reason=finish,
            done_marker=done,
            usage=usage,
            returned_output_tokens=len(token_ids),
            returned_prompt_ids_verified=prompt_ids == prompt if prompt_ids is not None else None,
            ttft_s=first - start if first is not None else None,
            tpot_s=(last - first) / (len(token_ids) - 1) if len(token_ids) > 1 and first is not None else None,
            end_to_end_s=end - start,
            end_monotonic_s=end,
            token_ids=token_ids,
            token_bursts=bursts,
            text="".join(text_parts),
        )
    except Exception as exc:
        body = exc.read().decode(errors="replace") if isinstance(exc, urllib.error.HTTPError) else None
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            http_error_body=body,
            returned_output_tokens=len(token_ids),
            token_ids=token_ids,
            token_bursts=bursts,
            end_to_end_s=time.perf_counter() - start,
            end_monotonic_s=time.perf_counter(),
        )
    return result


def main():
    """Run concurrent requests and report verified throughput with metric deltas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PRESETS, default="full")
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="lora1")
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--excluded-token-ids", default="0,1,128799")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--control-timeout", type=float, default=1800)
    parser.add_argument("--metrics-settle-seconds", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pin, pout, concurrency = PRESETS[args.mode]
    args.input_tokens = args.input_tokens or pin
    args.output_tokens = args.output_tokens or pout
    args.concurrency = args.concurrency or concurrency
    args.requests = args.requests or args.concurrency
    args.seed = args.seed if args.seed is not None else secrets.randbits(63)
    if min(args.input_tokens, args.output_tokens, args.concurrency, args.requests) <= 0:
        parser.error("Token counts, concurrency, and requests must be positive")
    if args.mode == "full" and (args.input_tokens, args.output_tokens, args.concurrency) != (8192, 4096, 8):
        parser.error("full mode fixes the required 8192/4096/C8 scenario; use smoke/profile for other shapes")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing directory {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    params = vars(args).copy()
    params["output_dir"] = str(args.output_dir)
    params["argv"] = sys.argv
    params["request_contract"] = {
        "endpoint": "/v1/completions",
        "ignore_eos": True,
        "add_special_tokens": False,
        "return_token_ids": True,
        "temperature": 0,
    }
    (args.output_dir / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    banned = {int(item) for item in args.excluded_token_ids.split(",") if item}
    models = json.loads(http(args.base_url, "/v1/models"))
    (args.output_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    if args.model not in {item["id"] for item in models.get("data", [])}:
        raise RuntimeError(f"Model {args.model} is absent from /v1/models")
    prompts = [generate_prompt(args.seed, i, args.input_tokens, args.vocab_size, banned) for i in range(args.requests)]
    if len({tuple(p[: min(32, len(p))]) for p in prompts}) != len(prompts):
        raise RuntimeError("Prompt prefix collision; choose another seed")
    (args.output_dir / "prompts.json").write_text(json.dumps(prompts, separators=(",", ":")) + "\n")
    for i in range(args.warmup_requests):
        original = args.output_tokens
        args.output_tokens = 16
        warm = request_once(args, -i - 1, generate_prompt(args.seed, -i - 1, 128, args.vocab_size, banned))
        args.output_tokens = original
        if warm["status"] != "success":
            raise RuntimeError(f"Warmup failed: {warm}")
    time.sleep(args.metrics_settle_seconds)
    before_text = http(args.base_url, "/metrics")
    (args.output_dir / "metrics-before.txt").write_text(before_text)
    profiling = False
    results = []
    try:
        if args.mode == "profile":
            http(args.base_url, "/start_profile", {}, args.control_timeout)
            profiling = True
        gate = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request_once, args, i, prompt, gate) for i, prompt in enumerate(prompts)]
            start = time.perf_counter()
            gate.set()
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                results.append(item)
                with (args.output_dir / "requests.jsonl").open("a") as log:
                    log.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(
                    f"request {item['index']}: {item['status']} "
                    f"tokens={item.get('returned_output_tokens')} latency={item['end_to_end_s']:.3f}s",
                    flush=True,
                )
            end = max(item["end_monotonic_s"] for item in results)
    finally:
        if profiling:
            (args.output_dir / "profile-stop-response.txt").write_text(
                http(args.base_url, "/stop_profile", {}, args.control_timeout)
            )
    time.sleep(args.metrics_settle_seconds)
    after_text = http(args.base_url, "/metrics")
    (args.output_dir / "metrics-after.txt").write_text(after_text)
    success = [item for item in results if item["status"] == "success"]
    elapsed = end - start
    generated = sum(item["returned_output_tokens"] for item in success)
    report = {
        "status": "PASS" if len(success) == args.requests else "FAIL",
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "input_tokens_per_request": args.input_tokens,
        "output_tokens_per_request": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "successful_requests": len(success),
        "measurement_wall_s": elapsed,
        "verified_output_tokens": generated,
        "effective_output_tokens_per_second": generated / elapsed,
        "ttft_s": distribution([item["ttft_s"] for item in success]),
        "tpot_ms": distribution([1000 * item["tpot_s"] for item in success if item["tpot_s"] is not None]),
        "latency_s": distribution([item["end_to_end_s"] for item in success]),
        "speculative_metrics": metric_delta(metrics(before_text), metrics(after_text)),
        "measurement_notes": [
            "Each request submits exact token IDs; no chat template or client retokenization.",
            "Independent random prefixes avoid intra-run prefix-cache reuse; "
            "repeats of a seed on the same live prefix-caching engine are invalid comparisons.",
            "TTFT uses first nonempty output-token-ID chunk; "
            "TPOT=(last token arrival-first token arrival)/(output_tokens-1).",
            "DSpark can return multiple tokens per chunk; chunk intervals are not per-token decode latency.",
            "Output throughput includes full prompt/prefill and all eight requests; "
            "only fully verified successful outputs count.",
            "Profile runs contain profiler overhead and are excluded from performance comparisons.",
            "Random token performance workload does not establish model/adapter output quality.",
        ],
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
