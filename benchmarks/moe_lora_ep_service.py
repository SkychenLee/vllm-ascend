# SPDX-License-Identifier: Apache-2.0
"""Repeat fixed-token streaming requests against a running MoE LoRA service.

Use this for the EP AllGather MoE LoRA service. Each request uses distinct
valid token IDs to avoid prefix cache hits.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


async def request(session, url, model, input_len, output_len, seed):
    rng = random.Random(seed)
    prompt = [rng.randrange(1000, 120000) for _ in range(input_len)]
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    token_times = []
    usage = None
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for raw_line in response.content:
            line = raw_line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                token_ids = choice.get("token_ids")
                if token_ids:
                    token_times.extend([time.perf_counter()] * len(token_ids))
                elif choice.get("text"):
                    token_times.append(time.perf_counter())
    if len(token_times) != output_len or usage is None or usage["completion_tokens"] != output_len:
        raise RuntimeError(f"incomplete streaming response: {len(token_times)} chunks, usage={usage}")
    ttft = token_times[0] - start
    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": (token_times[-1] - token_times[0]) * 1000 / max(len(token_times) - 1, 1),
        "total_ms": (time.perf_counter() - start) * 1000,
        "token_chunks": len(token_times),
        "usage": usage,
    }


async def run(args):
    url = args.base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    result = {"config": vars(args).copy(), "warmup": [], "rounds": []}
    result["config"]["output"] = str(args.output)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for phase, count in (("warmup", args.warmup), ("rounds", args.rounds)):
            for round_idx in range(count):
                samples = await asyncio.gather(
                    *(
                        request(
                            session,
                            url,
                            args.model,
                            args.input_len,
                            args.output_len,
                            args.seed
                            + round_idx * args.concurrency
                            + request_idx
                            + (0 if phase == "warmup" else 1000000),
                        )
                        for request_idx in range(args.concurrency)
                    )
                )
                summary = {
                    "samples": samples,
                    "median_ttft_ms": statistics.median(sample["ttft_ms"] for sample in samples),
                    "median_tpot_ms": statistics.median(sample["tpot_ms"] for sample in samples),
                    "max_tpot_ms": max(sample["tpot_ms"] for sample in samples),
                }
                result[phase].append(summary)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"{phase} {round_idx + 1}/{count}: {json.dumps(summary)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8778")
    parser.add_argument("--model", default="lora1")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.input_len, args.output_len, args.concurrency, args.rounds, args.timeout) <= 0 or args.warmup < 0:
        parser.error("lengths, concurrency, rounds and timeout must be positive; warmup must be nonnegative")
    if args.output.exists():
        parser.error("output exists; preserve prior measurements")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
