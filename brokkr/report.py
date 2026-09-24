"""Build the Brokkr results web page from saved JSON results.

Every number on the page is read from (or derived from) result files. Nothing is typed in by
hand, and if there are no results the page says so instead of showing placeholders.
"""

import datetime
import html
import json
from pathlib import Path

import numpy as np

from brokkr.accuracy import paired_bootstrap_diff
from brokkr.fingerprint import git_info

PRECISION_ORDER = {"fp32": 0, "fp16": 1, "int8": 2}
UNSTABLE_SPREAD_PCT = 10  # sessions disagreeing by more than this are flagged


def load_json_files(folder) -> list:
    folder = Path(folder)
    return [json.loads(p.read_text()) for p in sorted(folder.rglob("*.json"))] if folder.exists() else []


def order(record) -> tuple:
    return record["model"], PRECISION_ORDER.get(record["precision"], 99)


def top1_correct(record) -> np.ndarray:
    labels = np.array(record["raw"]["labels"])
    return (np.array(record["raw"]["top5_predictions"])[:, 0] == labels).astype(float)


def accuracy_rows(records: list) -> list:
    """One row per accuracy result, with the paired change vs FP32 on the same images."""
    by_key = {(r["model"], r["precision"], r["settings"]["dataset"], r["settings"]["n_images"]): r
              for r in records}
    rows = []
    for r in sorted(records, key=lambda r: (r["settings"]["n_images"],) + order(r)):
        s, m = r["settings"], r["metrics"]
        row = {"model": r["model"], "precision": r["precision"], "dataset": s["dataset"],
               "n_images": s["n_images"], "top1": m["top1"], "top1_ci95": m["top1_ci95"],
               "top5": m["top5"], "diff_vs_fp32": None}
        fp32 = by_key.get((r["model"], "fp32", s["dataset"], s["n_images"]))
        if fp32 and r["precision"] != "fp32" and fp32["raw"]["labels"] == r["raw"]["labels"]:
            row["diff_vs_fp32"] = paired_bootstrap_diff(top1_correct(fp32), top1_correct(r))
        rows.append(row)
    return rows


def speed_rows(records: list) -> list:
    rows = []
    # Group by power state first, so battery and plugged-in runs aren't mixed together.
    for r in sorted(records, key=lambda r: (str(r["machine"]["power"]["on_ac_power"]),) + order(r)
                    + (r["settings"]["num_threads"],)):
        m, power = r["metrics"], r["machine"]["power"]
        rows.append({
            "model": r["model"], "precision": r["precision"], "threads": r["settings"]["num_threads"],
            "p50_ms": m["p50_ms"], "p95_ms": m["p95_ms"], "p99_ms": m["p99_ms"],
            "spread_pct": m.get("p50_spread_pct"), "sessions": r["settings"].get("sessions", 1),
            "power": {True: "plugged in", False: "battery", None: "unknown"}[power["on_ac_power"]],
            "power_mode": power.get("power_mode") or "unknown",
            "cpu": r["machine"]["cpu_model"],
        })
    return rows


def size_rows(model_records: list) -> list:
    return [{"model": r["model"], "precision": r["precision"], "size_mb": r["file"]["size_bytes"] / 1e6}
            for r in sorted(model_records, key=order)]


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(accuracy: list, speed: list, sizes: list, machines: list, licences: dict) -> str:
    e = html.escape
    sections = []

    if sizes:
        sections.append("<h2>Model size</h2>" + table(
            ["Model", "Precision", "File size"],
            [[e(r["model"]), e(r["precision"]), f"{r['size_mb']:.1f} MB"] for r in sizes]))

    if accuracy:
        rows = []
        for r in accuracy:
            lo, hi = r["top1_ci95"]
            diff = "—"
            if r["diff_vs_fp32"]:
                d, dlo, dhi = r["diff_vs_fp32"]
                diff = f"{d * 100:+.2f} pts ({dlo * 100:+.2f} to {dhi * 100:+.2f})"
            rows.append([e(r["model"]), e(r["precision"]), e(r["dataset"]), f"{r['n_images']:,}",
                         f"{pct(r['top1'])} ({pct(lo)} – {pct(hi)})", pct(r["top5"]), diff])
        sections.append(
            "<h2>Accuracy</h2><p>Top-1 = the model's first guess is right. Brackets are 95% "
            "bootstrap confidence intervals. “vs FP32” is the paired difference on the same images.</p>"
            + table(["Model", "Precision", "Dataset", "Images", "Top-1 (95% CI)", "Top-5", "Top-1 vs FP32"],
                    rows))

    if speed:
        rows = []
        for r in speed:
            spread = "—" if r["spread_pct"] is None else f"{r['spread_pct']:.1f}%"
            if r["spread_pct"] is not None and r["spread_pct"] > UNSTABLE_SPREAD_PCT:
                spread += " ⚠ unstable"
            rows.append([e(r["model"]), e(r["precision"]), r["threads"], f"{r['p50_ms']:.2f}",
                         f"{r['p95_ms']:.2f}", f"{r['p99_ms']:.2f}", spread, r["sessions"],
                         e(f"{r['power']}, {r['power_mode']}"), e(r["cpu"])])
        sections.append(
            "<h2>Speed</h2><p>Time to classify one image, in milliseconds (median across sessions). "
            f"“Spread” is how much the sessions disagreed; above {UNSTABLE_SPREAD_PCT}% the machine was "
            "not stable and the numbers are rough.</p>"
            + table(["Model", "Precision", "Threads", "p50 ms", "p95 ms", "p99 ms", "Spread",
                     "Sessions", "Power", "CPU"], rows))

    if not sections:
        sections.append("<p>No results yet.</p>")

    if machines:
        rows = [[e(m["cpu_model"]), e(f"{m['os']} {m['os_release']}"),
                 e(str(m["packages"].get("onnxruntime")))] for m in machines]
        sections.append("<h2>Machines</h2>" + table(["CPU", "OS", "ONNX Runtime"], rows))

    if licences:
        items = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in licences.items())
        sections.append(f"<h2>Licences</h2><ul>{items}</ul>")

    commit = git_info()["commit"] or "unknown"
    built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brokkr results</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid #999; padding: 0.3rem 0.6rem; text-align: left; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Brokkr: measured results</h1>
<p>Early development (Stage 1 of 7). Every number below was measured by Brokkr's own code on the
machine listed; nothing is estimated or copied from elsewhere.</p>
{"".join(sections)}
<hr>
<p><small>Built {built} from code at commit {e(commit[:7])}.
Source: <a href="https://github.com/Vasco1290/Brokkr">github.com/Vasco1290/Brokkr</a></small></p>
</body>
</html>
"""


def build_site(results_dir, models_dir, out_dir) -> Path:
    """Read every result file and write out_dir/index.html."""
    results = load_json_files(results_dir)
    accuracy = [r for r in results if r.get("kind") == "accuracy" and r.get("source") == "brokkr"]
    speed = [r for r in results if r.get("kind") == "speed" and r.get("source") == "brokkr"]
    model_records = [r for r in load_json_files(models_dir) if "file" in r]

    machines = {r["machine"]["cpu_model"] + r["machine"]["os"]: r["machine"] for r in accuracy + speed}
    licences = {}
    for r in model_records:
        licences[f"{r['model']} weights"] = r["licence"]["weights"]
        licences[f"{r['model']} code"] = r["licence"]["code"]
    for r in accuracy:
        licences[f"{r['settings']['dataset']} dataset"] = r["settings"]["dataset_licence"]

    page = render_html(accuracy_rows(accuracy), speed_rows(speed), size_rows(model_records),
                       list(machines.values()), licences)
    out = Path(out_dir) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
