"""Build the Brokkr results web page from saved JSON results.

Every number on the page is read from (or derived from) result files. Nothing is typed in by
hand, and if there are no results the page says so instead of showing placeholders.
"""

import datetime
import html
import json
from pathlib import Path

import numpy as np

from brokkr import charts
from brokkr.accuracy import paired_bootstrap_diff
from brokkr.fingerprint import git_info

PRECISION_ORDER = {"fp32": 0, "fp16": 1, "int8": 2}
UNSTABLE_IQR_PCT = 10  # middle half of sessions disagreeing by more than this is flagged


def load_json_files(folder) -> list:
    folder = Path(folder)
    return [json.loads(p.read_text()) for p in sorted(folder.rglob("*.json"))] if folder.exists() else []


def order(record) -> tuple:
    return record["model"], PRECISION_ORDER.get(record["precision"], 99)


def top1_correct(record) -> np.ndarray:
    labels = np.array(record["raw"]["labels"])
    return (np.array(record["raw"]["top5_predictions"])[:, 0] == labels).astype(float)


def accuracy_rows(records: list) -> list:
    """One row per accuracy result, with the paired change vs FP32 on the same images.

    Only the test split and the full dataset are shown; calibration/tuning splits are inputs
    to other measurements, not results in their own right.
    """
    records = [r for r in records if r["settings"].get("split", "test") in ("test", "all")]
    by_key = {(r["model"], r["precision"], r["settings"]["dataset"], r["settings"]["n_images"]): r
              for r in records}
    rows = []
    for r in sorted(records, key=lambda r: (r["settings"]["n_images"],) + order(r)):
        s, m = r["settings"], r["metrics"]
        row = {"model": r["model"], "precision": r["precision"], "dataset": s["dataset"],
               "n_images": s["n_images"], "top1": m["top1"], "top1_ci95": m["top1_ci95"],
               "top5": m["top5"], "diff_vs_fp32": None, "ties": m.get("top1_tied_images", 0)}
        fp32 = by_key.get((r["model"], "fp32", s["dataset"], s["n_images"]))
        if fp32 and r["precision"] != "fp32" and fp32["raw"]["labels"] == r["raw"]["labels"]:
            row["diff_vs_fp32"] = paired_bootstrap_diff(top1_correct(fp32), top1_correct(r))
        rows.append(row)
    return rows


def speed_rows(records: list) -> list:
    rows = []
    # Group by power state first, so battery and plugged-in runs aren't mixed together.
    for r in sorted(records, key=lambda r: (str(r["machine"]["power"]["on_ac_power"]),
                                           r["settings"].get("cores", "not pinned")) + order(r)
                    + (r["settings"]["num_threads"],)):
        m, power = r["metrics"], r["machine"]["power"]
        rows.append({
            "model": r["model"], "precision": r["precision"], "threads": r["settings"]["num_threads"],
            "p50_ms": m["p50_ms"], "p95_ms": m["p95_ms"], "p99_ms": m["p99_ms"],
            "spread_pct": m.get("p50_spread_pct"), "iqr_pct": m.get("p50_iqr_pct"),
            "sessions": r["settings"].get("sessions", 1),
            "cores": r["settings"].get("cores", "not pinned"),
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


CORRUPTION_ORDER = ["fog", "defocus_blur", "motion_blur", "noise", "darkness"]


def robustness_html(sweep: list, reliability: list) -> str:
    """Charts and tables for the corruption sweep (Stage 2). Empty string if there is no sweep."""
    if not sweep:
        return ""
    e = html.escape
    acc = {(r["precision"], r["settings"]["corruption"], r["settings"]["severity"]): r["metrics"]
           for r in sweep}
    rel = {(r["kind"], r["precision"], r["settings"]["corruption"], r["settings"]["severity"]): r["metrics"]
           for r in reliability if "corruption" in r["settings"]}
    precisions = [p for p in PRECISION_ORDER if any(k[0] == p for k in acc)]
    corruptions = [c for c in CORRUPTION_ORDER if any(k[1] == c for k in acc)]
    severities = [0, 1, 2, 3, 4, 5]

    def value(key_of, getter):
        """Series per precision for one corruption; severity 0 is the clean condition."""
        def at(p, c, s):
            return getter(key_of(p, "clean" if s == 0 else c, s))
        return {c: {p: [at(p, c, s) for s in severities] for p in precisions} for c in corruptions}

    top1 = value(lambda p, c, s: acc[(p, c, s)], lambda m: m["top1"] * 100)
    cover = value(lambda p, c, s: rel[("conformal", p, c, s)], lambda m: m["coverage"] * 100)
    sizes = value(lambda p, c, s: rel[("conformal", p, c, s)], lambda m: m["mean_set_size"])
    size_max = max(v for c in sizes.values() for vals in c.values() for v in vals)
    size_max = float(int(size_max / 5) * 5 + 5)  # round up to a multiple of 5

    def row_of(title, data, **kw):
        return ('<div class="panels">' + "".join(charts.panel(c.replace("_", " "), data[c], severities, **kw)
                                                  for c in corruptions) + "</div>")

    n = sweep[0]["settings"]["n_images"]
    out = [
        "<h2>Robustness: damaged images</h2>",
        f"<p>The same {n:,} test images, damaged by five kinds of simulated bad camera conditions at "
        "severity 1 (mild) to 5 (severe); severity 0 is the clean image. Each image gets the same damage "
        "pattern for every precision. Conformal prediction sets were tuned for 90% coverage on separate "
        "clean images, so the second row shows whether that promise survives damage it was not tuned for. "
        "FP16 and FP32 overlap almost exactly (FP32's circle sits on FP16's square). "
        "Hover a point for its value; every number is also in the table below.</p>",
        charts.legend(precisions),
        "<h3>Top-1 accuracy</h3>", row_of("Top-1 accuracy", top1),
        "<h3>Conformal coverage (target 90%)</h3>",
        row_of("Coverage", cover, reference=90, reference_label="90% target"),
        "<h3>Average prediction-set size</h3><p>How many classes the model offers in its set. If the model "
        "noticed it was struggling, sets would grow as damage gets worse.</p>",
        row_of("Set size", sizes, y_max=size_max, unit=""),
    ]

    headers = ["Condition", "Severity"]
    for name in ("Top-1", "Coverage", "Set size", "ECE"):
        headers += [f"{name} {p.upper()}" for p in precisions]
    rows = []
    for c in ["clean"] + corruptions:
        for s in ([0] if c == "clean" else severities[1:]):
            row = [e(c.replace("_", " ")), s]
            row += [pct(acc[(p, c, s)]["top1"]) for p in precisions]
            row += [pct(rel[("conformal", p, c, s)]["coverage"]) for p in precisions]
            row += [f"{rel[('conformal', p, c, s)]['mean_set_size']:.2f}" for p in precisions]
            row += [f"{rel[('calibration', p, c, s)]['ece']:.3f}" for p in precisions]
            rows.append(row)
    out.append("<h3>All numbers</h3><p>ECE (expected calibration error) measures how far the model's "
               "confidence is from its actual accuracy; 0 is perfectly honest. Small ECE values are biased "
               "upwards, so don't over-read differences below about 0.02.</p>" + table(headers, rows))

    summary = []
    for p in precisions:
        cal, conf, sel = (rel[(k, p, "clean", 0)] for k in ("calibration", "conformal", "selective"))
        summary.append([e(p.upper()), f"{cal['ece']:.3f}", f"{cal['overconfidence'] * 100:+.1f} pts",
                        f"{conf['coverage'] * 100:.2f}%", f"{conf['mean_set_size']:.2f}",
                        f"{sel['aurc']:.3f}", f"{sel['e_aurc']:.3f}",
                        f"{sel['risk_at_50pct_coverage'] * 100:.1f}%"])
    out.append("<h3>Knowing when it's wrong, on clean images</h3><p>Overconfidence = average confidence "
               "minus accuracy (negative = under-confident). AURC: error averaged over answering only the "
               "most confident 1%…100% of images (lower is better); E-AURC removes the part due to accuracy "
               "alone. Last column: error rate when answering only the most confident half.</p>"
               + table(["Precision", "ECE", "Overconfidence", "Coverage", "Set size", "AURC", "E-AURC",
                        "Error at 50% answered"], summary))
    return "".join(out)


def render_html(accuracy: list, speed: list, sizes: list, machines: list, licences: dict,
                robustness: str = "") -> str:
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
            top1 = f"{pct(r['top1'])} ({pct(lo)} – {pct(hi)})"
            if r["ties"]:
                top1 += f" · {r['ties']:,} tied"
            rows.append([e(r["model"]), e(r["precision"]), e(r["dataset"]), f"{r['n_images']:,}",
                         top1, pct(r["top5"]), diff])
        sections.append(
            "<h2>Accuracy</h2><p>Top-1 = the model's first guess is right. Brackets are 95% "
            "bootstrap confidence intervals. “vs FP32” is the paired difference on the same images. "
            "“Tied” counts images where two classes had exactly the same top score (common for INT8, "
            "whose outputs are rounded); ties go to the lower class number.</p>"
            + table(["Model", "Precision", "Dataset", "Images", "Top-1 (95% CI)", "Top-5", "Top-1 vs FP32"],
                    rows))

    if robustness:
        sections.append(robustness)

    if speed:
        rows = []
        for r in speed:
            spread = "—" if r["spread_pct"] is None else f"{r['spread_pct']:.1f}%"
            iqr = "—" if r["iqr_pct"] is None else f"{r['iqr_pct']:.1f}%"
            if r["iqr_pct"] is not None and r["iqr_pct"] > UNSTABLE_IQR_PCT:
                iqr += " ⚠ unstable"
            rows.append([e(r["model"]), e(r["precision"]), r["threads"], e(r["cores"]), f"{r['p50_ms']:.2f}",
                         f"{r['p95_ms']:.2f}", f"{r['p99_ms']:.2f}", iqr, spread, r["sessions"],
                         e(f"{r['power']}, {r['power_mode']}"), e(r["cpu"])])
        sections.append(
            "<h2>Speed</h2><p>Time to classify one image, in milliseconds (median across sessions). "
            "“IQR” is how much the middle half of sessions disagreed; above "
            f"{UNSTABLE_IQR_PCT}% the machine was not stable and the numbers are rough. “Full spread” "
            "compares the fastest and slowest session, so one unlucky session makes it large.</p>"
            + table(["Model", "Precision", "Threads", "Cores", "p50 ms", "p95 ms", "p99 ms", "IQR",
                     "Full spread", "Sessions", "Power", "CPU"], rows))

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
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --ink: #0b0b0b; --ink-secondary: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --border: #c3c2b7;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface: #1a1a19; --ink: #ffffff; --ink-secondary: #c3c2b7; --ink-muted: #898781;
      --grid: #2c2c2a; --border: #383835;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    }}
  }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 1180px;
          margin: 2rem auto; padding: 0 16px; background: var(--surface); color: var(--ink); }}
  p {{ color: var(--ink-secondary); max-width: 75ch; }}
  a {{ color: var(--series-1); }}
  table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; display: block; overflow-x: auto;
           max-width: 100%; font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid var(--border); padding: 0.3rem 0.6rem; text-align: left; white-space: nowrap; }}
  .panels {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .panel {{ width: 220px; max-width: 100%; height: auto; }}
  .panel-title {{ font-size: 12px; font-weight: 600; fill: var(--ink); }}
  .tick {{ font-size: 10px; fill: var(--ink-muted); font-variant-numeric: tabular-nums; }}
  .legend {{ display: flex; gap: 16px; font-size: 0.9rem; margin: 0.5rem 0; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
</style>
</head>
<body>
<h1>Brokkr: measured results</h1>
<p>Early development (see ROADMAP.md for progress). Every number below was measured by
Brokkr's own code on the machine listed; nothing is estimated or copied from elsewhere.</p>
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
    ours = [r for r in results if r.get("source") == "brokkr"]
    accuracy = [r for r in ours if r["kind"] == "accuracy" and "corruption" not in r["settings"]]
    # Sweeps on the tuning/calibration splits are inputs for choosing Stage 3 settings, not results.
    sweep = [r for r in ours if r["kind"] == "accuracy" and "corruption" in r["settings"]
             and r["settings"].get("split", "test") == "test"]
    reliability = [r for r in ours if r["kind"] in ("calibration", "conformal", "selective")]
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
                       list(machines.values()), licences, robustness_html(sweep, reliability))
    out = Path(out_dir) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
