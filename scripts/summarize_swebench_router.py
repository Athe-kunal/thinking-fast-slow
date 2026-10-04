"""Summarizes a `swebench_router_experiment` run.

Per router setting: resolve rate, non-empty patches, exit statuses, agent steps,
format errors, and decoding statistics from the model servers' request logs
(share of tokens decoded by diffusion, tokens per forward pass, speed).

    uv run python -m scripts.summarize_swebench_router <run_dir>
"""

import argparse
import collections
import json
import pathlib
import re

SEGMENT = re.compile(r"(ar|dlm)(\d+)")


def instance_rows(setting_dir: pathlib.Path, resolved: dict) -> list[dict]:
    """Returns one row per finished instance of a setting."""
    rows = []
    for traj in sorted(setting_dir.glob("lane*/*/*.traj.json")):
        data = json.loads(traj.read_text())
        info = data["info"]
        iid = data["instance_id"]
        messages = data["messages"]
        format_errors = sum(
            1
            for m in messages
            if m.get("role") == "user"
            and "format error" in str(m.get("content", "")).lower()
        )
        rows.append(
            {
                "instance_id": iid,
                "resolved": resolved.get(iid),
                "submitted": bool((info.get("submission") or "").strip()),
                "exit_status": info.get("exit_status"),
                "steps": info.get("model_stats", {}).get("api_calls", 0),
                "format_errors": format_errors,
            }
        )
    return rows


def decode_stats(setting_dir: pathlib.Path) -> dict:
    """Aggregates the model servers' per-request logs of a setting."""
    tokens = {"ar": 0, "dlm": 0}
    nfe = seconds = prompt = n = 0
    for path in setting_dir.glob("lane*/requests.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            n += 1
            nfe += r["nfe"]
            seconds += r["seconds"]
            prompt += r["prompt_tokens"]
            for mode, count in SEGMENT.findall(r.get("segments", "")):
                tokens[mode] += int(count)
    total = tokens["ar"] + tokens["dlm"]
    return {
        "requests": n,
        "completion_tokens": total,
        "dlm_token_share": tokens["dlm"] / total if total else 0.0,
        "tokens_per_forward": total / nfe if nfe else 0.0,
        "tokens_per_second": total / seconds if seconds else 0.0,
        "mean_prompt_tokens": prompt / n if n else 0.0,
        "generation_hours": seconds / 3600,
    }


def main() -> None:
    """Prints the summary table and writes summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()

    results_path = args.run_dir / "results.json"
    results = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    summary = {}
    for setting_dir in sorted(args.run_dir.glob("p*")):
        setting = setting_dir.name
        resolved = {
            iid: ok
            for wave in results.get(setting, {}).values()
            for iid, ok in wave.items()
        }
        rows = instance_rows(setting_dir, resolved)
        scored = [r for r in rows if r["resolved"] is not None]
        n = len(rows)
        summary[setting] = {
            "finished": n,
            "scored": len(scored),
            "resolved": sum(bool(r["resolved"]) for r in scored),
            "submitted": sum(r["submitted"] for r in rows),
            "exit_statuses": dict(
                collections.Counter(r["exit_status"] for r in rows)
            ),
            "mean_steps": sum(r["steps"] for r in rows) / n if n else 0.0,
            "instances_with_format_errors": sum(
                r["format_errors"] > 0 for r in rows
            ),
            "format_errors_per_step": (
                sum(r["format_errors"] for r in rows)
                / max(1, sum(r["steps"] for r in rows))
            ),
            **decode_stats(setting_dir),
            "instances": rows,
        }

    print(
        f"{'setting':8} {'resolved':>9} {'patch':>9} {'steps':>6} "
        f"{'fmt-err/step':>12} {'dlm-tok':>8} {'tok/fwd':>8} {'tok/s':>6}"
    )
    for setting, s in summary.items():
        print(
            f"{setting:8} {s['resolved']:>4}/{s['scored']:<4} "
            f"{s['submitted']:>4}/{s['finished']:<4} {s['mean_steps']:6.1f} "
            f"{s['format_errors_per_step']:12.2f} "
            f"{s['dlm_token_share']:8.1%} {s['tokens_per_forward']:8.2f} "
            f"{s['tokens_per_second']:6.1f}"
        )
        print(f"         exit statuses: {s['exit_statuses']}")
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
