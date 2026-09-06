#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare single-stream EP/non-EP and zero-adapter outputs on identical prompts.

Run against each server separately with --label noep / ep, then use --compare.
The input is an existing first-token report: its prompts/labels are reused exactly.
Invalid answers count as incorrect; missing logprobs are errors, never a pass.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = "Answer the statement with exactly one word: TRUE if it is true, otherwise FALSE. Do not explain."
TRUE_TOKENS = {"T", "TRUE"}
FALSE_TOKENS = {"F", "FALSE"}


def normalized_answer(token):
    normalized = token.strip().upper()
    if normalized in TRUE_TOKENS:
        return True
    if normalized in FALSE_TOKENS:
        return False
    return None


def ask(base_url, model, question, max_tokens, system_prompt=SYSTEM_PROMPT):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        decoded = json.load(response)
    choice = decoded["choices"][0]
    logprobs = (choice.get("logprobs") or {}).get("content") or []
    if not logprobs:
        raise RuntimeError(f"Missing output logprobs: {decoded}")
    if any(not math.isfinite(item["logprob"]) for item in logprobs):
        raise RuntimeError(f"Non-finite selected logprob: {decoded}")
    return {
        "model": model,
        "text": choice["message"].get("content"),
        "reasoning": choice["message"].get("reasoning"),
        "logprobs": logprobs,
        "finish_reason": choice.get("finish_reason"),
        "usage": decoded.get("usage"),
        "elapsed_s": time.monotonic() - started,
    }


def pair_metrics(pairs):
    selected_equal = 0
    sequence_equal = 0
    first_deltas = []
    candidate_deltas = []
    prefix_probability_deltas = []
    common_prefix_lengths = []
    for left, right in pairs:
        left_first, right_first = left["logprobs"][0], right["logprobs"][0]
        selected_equal += left_first["token"] == right_first["token"]
        sequence_equal += [x["token"] for x in left["logprobs"]] == [x["token"] for x in right["logprobs"]]
        common_prefix = 0
        for a, b in zip(left["logprobs"], right["logprobs"]):
            if a["token"] != b["token"]:
                break
            common_prefix += 1
            prefix_probability_deltas.append(abs(a["logprob"] - b["logprob"]))
        common_prefix_lengths.append(common_prefix)
        if left_first["token"] == right_first["token"]:
            first_deltas.append(abs(left_first["logprob"] - right_first["logprob"]))
        left_candidates = {x["token"]: x["logprob"] for x in left_first["top_logprobs"]}
        right_candidates = {x["token"]: x["logprob"] for x in right_first["top_logprobs"]}
        for token in left_candidates.keys() & right_candidates.keys():
            candidate_deltas.append(abs(left_candidates[token] - right_candidates[token]))
    return {
        "pairs": len(pairs),
        "same_first_token": selected_equal,
        "same_token_sequence": sequence_equal,
        "common_prefix_tokens_mean": sum(common_prefix_lengths) / len(pairs) if pairs else None,
        "common_prefix_logprob_max_abs": max(prefix_probability_deltas, default=None),
        "same_token_logprob_max_abs": max(first_deltas, default=None),
        "shared_candidate_logprob_max_abs": max(candidate_deltas, default=None),
        "shared_candidate_logprob_mean_abs": sum(candidate_deltas) / len(candidate_deltas)
        if candidate_deltas
        else None,
    }


def summarize(rows, score_mode="binary_first_token"):
    report = {}
    for label in rows[0]["models"]:
        if score_mode == "none":
            report[label] = {
                "total": len(rows),
                "output_tokens": sum(len(row["models"][label]["logprobs"]) for row in rows),
            }
            continue
        answers = [normalized_answer(row["models"][label]["logprobs"][0]["token"]) for row in rows]
        correct = sum(answer is not None and answer == row["answer"] for answer, row in zip(answers, rows, strict=True))
        report[label] = {
            "total": len(rows),
            "valid": sum(answer is not None for answer in answers),
            "correct": correct,
            "accuracy": correct / len(rows),
        }
    for label in rows[0]["models"]:
        if label != "ds" and "ds" in rows[0]["models"]:
            report[f"ds_vs_{label}"] = pair_metrics([(row["models"]["ds"], row["models"][label]) for row in rows])
    return report


def compare_reports(paths, out):
    left, right = [json.loads(Path(path).read_text()) for path in paths]
    for key in ("question_sha256", "system_prompt", "max_tokens", "workers", "model_specs", "score_mode"):
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Comparison configuration mismatch: {key}")
    result = {"left": left["config"], "right": right["config"], "models": {}}
    for model in left["rows"][0]["models"]:
        pairs = []
        for a, b in zip(left["rows"], right["rows"], strict=True):
            if (a["question"], a["answer"]) != (b["question"], b["answer"]):
                raise ValueError("Question/answer alignment mismatch")
            pairs.append((a["models"][model], b["models"][model]))
        result["models"][model] = pair_metrics(pairs)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--questions-from-report")
    parser.add_argument("--label")
    parser.add_argument("--models", default="ds=ds,lora1=lora1,ds_repeat=ds,news2026=news2026")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--score-mode", choices=["binary_first_token", "none"], default="binary_first_token")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.compare:
        compare_reports(args.compare, args.out)
        return
    if not args.questions_from_report or not args.label:
        parser.error("--questions-from-report and --label are required for evaluation")
    source = json.loads(Path(args.questions_from_report).read_text())
    rows = [
        {key: row[key] for key in ("topic", "question", "answer", "quote") if key in row}
        for row in source["rows"][: args.limit]
    ]
    if not rows:
        parser.error("No questions")
    model_specs = dict(spec.split("=", 1) for spec in args.models.split(","))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    config = {
        "label": args.label,
        "base_url": args.base_url,
        "question_sha256": digest,
        "system_prompt": args.system_prompt,
        "score_mode": args.score_mode,
        "model_specs": model_specs,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
    }
    jobs = [(i, label, model, row["question"]) for i, row in enumerate(rows) for label, model in model_specs.items()]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["models"] = {}

    def run(job):
        index, label, model, question = job
        return index, label, ask(args.base_url, model, question, args.max_tokens, args.system_prompt)

    # Persist each completed response, including all top logprobs, for diagnosis.
    with output.with_suffix(".jsonl").open("w") as journal:
        journal.write(json.dumps({"config": config}) + "\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for completed, (index, label, response) in enumerate(executor.map(run, jobs), 1):
                rows[index]["models"][label] = response
                journal.write(
                    json.dumps(
                        {"index": index, "label": label, "question": rows[index]["question"], "response": response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                journal.flush()
                if completed % 20 == 0 or completed == len(jobs):
                    print(f"{args.label}: {completed}/{len(jobs)}", flush=True)
    summary = summarize(rows, args.score_mode)
    output.write_text(json.dumps({"config": config, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
