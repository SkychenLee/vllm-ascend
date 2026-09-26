# SPDX-License-Identifier: Apache-2.0
"""Alternate identical MoE LoRA request batches between two eight-card services."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import aiohttp
from moe_lora_ep_service import request


async def batch(session, url, model, input_len, output_len, concurrency, seed):
    samples = await asyncio.gather(
        *(
            request(session, url.rstrip("/") + "/v1/completions", model, input_len, output_len, seed + index)
            for index in range(concurrency)
        )
    )
    return {
        "samples": samples,
        "median_ttft_ms": statistics.median(sample["ttft_ms"] for sample in samples),
        "median_tpot_ms": statistics.median(sample["tpot_ms"] for sample in samples),
        "max_ttft_ms": max(sample["ttft_ms"] for sample in samples),
        "max_tpot_ms": max(sample["tpot_ms"] for sample in samples),
    }


async def run(args):
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "warmup": [],
        "rounds": [],
    }
    services = {
        "candidate": (args.candidate_url, args.candidate_model),
        "baseline": (args.baseline_url, args.baseline_model),
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        for phase, count in (("warmup", args.warmup), ("rounds", args.rounds)):
            for round_idx in range(count):
                seed = args.seed + round_idx * args.concurrency + (0 if phase == "warmup" else 1000000)
                order = ("candidate", "baseline") if round_idx % 2 == 0 else ("baseline", "candidate")
                report = {"order": order, "seed": seed}
                for name in order:
                    url, model = services[name]
                    report[name] = await batch(
                        session, url, model, args.input_len, args.output_len, args.concurrency, seed
                    )
                result[phase].append(report)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"{phase} {round_idx + 1}/{count}: "
                    f"candidate TPOT={report['candidate']['median_tpot_ms']:.3f} ms, "
                    f"baseline TPOT={report['baseline']['median_tpot_ms']:.3f} ms, "
                    f"candidate TTFT={report['candidate']['median_ttft_ms']:.1f} ms, "
                    f"baseline TTFT={report['baseline']['median_ttft_ms']:.1f} ms",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8778")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8779")
    parser.add_argument("--candidate-model", default="lora1")
    parser.add_argument("--baseline-model", default="lora1")
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
