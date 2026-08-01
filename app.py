from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from src.evaluation.fitness import FitnessConfig
from src.ga.engine import GAConfig
from src.io.generate_data import generate_rules, save_rules_json
from src.main import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "outputs" / "web_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_RULE_EXTS = {".json", ".csv"}


app = Flask(__name__)
# Used only for UI messages; no sensitive usage.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "fw-ga-" + uuid.uuid4().hex


# In-memory run state for the web UI.
# (For a student demo, this is more than enough.)
RUNS: Dict[str, Dict[str, Any]] = {}


def _new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _safe_int(val: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(val)
        return max(lo, min(hi, x))
    except Exception:
        return default


def _safe_float(val: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(val)
        return max(lo, min(hi, x))
    except Exception:
        return default


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_improvements(metrics: dict) -> dict:
    """Derive a few friendly 'demo' numbers."""
    try:
        o = metrics["original"]
        g = metrics["ga_best"]

        speedup = (o["avg_checks"] / g["avg_checks"]) if g["avg_checks"] > 0 else None
        shadow_drop = (o["shadow_rate"] - g["shadow_rate"])  # absolute drop
        return {
            "speedup_x": speedup,
            "shadow_rate_drop": shadow_drop,
        }
    except Exception:
        return {"speedup_x": None, "shadow_rate_drop": None}


def _run_job(run_id: str) -> None:
    job = RUNS[run_id]
    out_dir = Path(job["out_dir"])
    job["state"] = "running"
    job["started_at"] = time.time()

    try:
        run_pipeline(
            rules_path=Path(job["rules_path"]),
            out_dir=out_dir,
            n_packets=int(job["n_packets"]),
            seed=int(job["seed"]),
            ga_cfg=GAConfig(**job["ga_cfg"]),
            fit_cfg=FitnessConfig(**job["fit_cfg"]),
            student_name=str(job["student_name"]),
        )

        metrics_path = out_dir / "metrics.json"
        metrics = _load_json(metrics_path) or {}
        job["metrics"] = metrics
        job["improvements"] = _compute_improvements(metrics)
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["finished_at"] = time.time()


def _list_recent_runs(limit: int = 8) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = _load_json(d / "metrics.json")
        runs.append(
            {
                "run_id": d.name,
                "has_metrics": m is not None,
                "metrics": m,
                "improvements": _compute_improvements(m) if m else None,
                "mtime": d.stat().st_mtime,
            }
        )
    runs.sort(key=lambda x: x["mtime"], reverse=True)
    return runs[:limit]


@app.get("/")
def index():
    recent = _list_recent_runs(limit=8)
    return render_template(
        "index.html",
        recent_runs=recent,
        defaults={
            "student": "Your Name",
            "mode": "demo",
            "generate_rules": 300,
            "packets": 3000,
            "seed": 0,
            "pop": 40,
            "gens": 40,
            "crossover": 0.90,
            "p_swap": 0.20,
            "p_inv": 0.10,
            "min_coverage": 0.990,
        },
    )


@app.post("/start")
def start():
    run_id = _new_run_id()
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Presets (nice for demos) ---
    mode = request.form.get("mode", "demo")
    presets = {
        "demo": {"pop": 40, "gens": 40, "packets": 3000, "generate_rules": 300},
        "balanced": {"pop": 60, "gens": 60, "packets": 5000, "generate_rules": 500},
        "quality": {"pop": 90, "gens": 100, "packets": 8000, "generate_rules": 1000},
    }
    p = presets.get(mode, presets["demo"])

    # --- Inputs ---
    student_name = request.form.get("student", "Your Name")
    seed = _safe_int(request.form.get("seed", 0), default=0, lo=0, hi=10_000_000)
    n_packets = _safe_int(request.form.get("packets", p["packets"]), default=p["packets"], lo=200, hi=200_000)

    # GA params
    pop = _safe_int(request.form.get("pop", p["pop"]), default=p["pop"], lo=10, hi=500)
    gens = _safe_int(request.form.get("gens", p["gens"]), default=p["gens"], lo=5, hi=2000)
    crossover = _safe_float(request.form.get("crossover", 0.90), default=0.90, lo=0.0, hi=1.0)
    p_swap = _safe_float(request.form.get("p_swap", 0.20), default=0.20, lo=0.0, hi=1.0)
    p_inv = _safe_float(request.form.get("p_inv", 0.10), default=0.10, lo=0.0, hi=1.0)
    min_coverage = _safe_float(request.form.get("min_coverage", 0.990), default=0.990, lo=0.0, hi=1.0)

    # Rules source
    rules_source = request.form.get("rules_source", "sample")
    rules_path: Path

    if rules_source == "upload":
        f = request.files.get("rules_file")
        if not f or not f.filename:
            RUNS[run_id] = {"state": "error", "error": "No rules file uploaded.", "out_dir": str(out_dir)}
            return redirect(url_for("run_view", run_id=run_id))

        filename = secure_filename(f.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_RULE_EXTS:
            RUNS[run_id] = {
                "state": "error",
                "error": "Rules file must be .json or .csv",
                "out_dir": str(out_dir),
            }
            return redirect(url_for("run_view", run_id=run_id))

        rules_path = out_dir / f"uploaded_rules{ext}"
        f.save(rules_path)

    elif rules_source == "generate":
        n_rules = _safe_int(
            request.form.get("generate_rules", p["generate_rules"]),
            default=p["generate_rules"],
            lo=20,
            hi=50_000,
        )
        rules = generate_rules(n_rules=n_rules, seed=seed)
        rules_path = out_dir / "generated_rules.json"
        save_rules_json(rules, rules_path)

    else:
        # Use built-in sample rules file
        rules_path = BASE_DIR / "data" / "rules_sample.json"
        if not rules_path.exists():
            # If the sample is missing, generate a small one
            rules = generate_rules(n_rules=p["generate_rules"], seed=seed)
            rules_path.parent.mkdir(parents=True, exist_ok=True)
            save_rules_json(rules, rules_path)

    RUNS[run_id] = {
        "state": "queued",
        "created_at": time.time(),
        "out_dir": str(out_dir),
        "rules_path": str(rules_path),
        "rules_source": rules_source,
        "student_name": student_name,
        "seed": seed,
        "n_packets": n_packets,
        "ga_cfg": {
            "population_size": pop,
            "generations": gens,
            "crossover_rate": crossover,
            "p_swap": p_swap,
            "p_inversion": p_inv,
        },
        "fit_cfg": {
            "min_coverage": min_coverage,
        },
    }

    t = threading.Thread(target=_run_job, args=(run_id,), daemon=True)
    t.start()
    RUNS[run_id]["thread_started"] = True

    return redirect(url_for("run_view", run_id=run_id))


@app.get("/run/<run_id>")
def run_view(run_id: str):
    job = RUNS.get(run_id)
    if job is None:
        # Maybe it's an older run on disk
        out_dir = RUNS_DIR / run_id
        if out_dir.exists():
            metrics = _load_json(out_dir / "metrics.json")
            job = {
                "state": "done" if metrics else "unknown",
                "out_dir": str(out_dir),
                "metrics": metrics,
                "improvements": _compute_improvements(metrics) if metrics else None,
            }
        else:
            return render_template("error.html", title="Run not found", message="Unknown run id."), 404

    # Friendly links for the template
    out_dir = Path(job["out_dir"])
    files = {
        "report_pdf": (out_dir / "report.pdf") if (out_dir / "report.pdf").exists() else None,
        "metrics": (out_dir / "metrics.json") if (out_dir / "metrics.json").exists() else None,
        "ordering": (out_dir / "ordering_ga_best.json") if (out_dir / "ordering_ga_best.json").exists() else None,
        "rules_ga": (out_dir / "rules_ga_ordered.json") if (out_dir / "rules_ga_ordered.json").exists() else None,
    }
    figs = []
    fig_dir = out_dir / "figures"
    if fig_dir.exists():
        for name in [
            "fitness_evolution.png",
            "avg_checks_before_after.png",
            "shadow_rate_before_after.png",
            "coverage_shadow.png",
        ]:
            p = fig_dir / name
            if p.exists():
                figs.append(name)

    return render_template(
        "run.html",
        run_id=run_id,
        job=job,
        files=files,
        figures=figs,
    )


@app.get("/status/<run_id>")
def status(run_id: str):
    job = RUNS.get(run_id)
    if job is None:
        out_dir = RUNS_DIR / run_id
        if out_dir.exists() and (out_dir / "metrics.json").exists():
            return jsonify({"state": "done"})
        return jsonify({"state": "unknown"}), 404
    payload = {
        "state": job.get("state", "unknown"),
        "error": job.get("error"),
    }
    return jsonify(payload)


@app.get("/runs/<run_id>/file/<path:filename>")
def run_file(run_id: str, filename: str):
    # Serve any file inside the run folder.
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        return render_template("error.html", title="Run not found", message="Unknown run id."), 404

    # Prevent path traversal
    target = (out_dir / filename).resolve()
    if out_dir.resolve() not in target.parents and target != out_dir.resolve():
        return render_template("error.html", title="Bad request", message="Invalid filename."), 400
    if not target.exists():
        return render_template("error.html", title="Not found", message="File does not exist."), 404

    # Use send_file for arbitrary path
    return send_file(target, as_attachment=False)


@app.get("/runs/<run_id>/figure/<name>")
def run_figure(run_id: str, name: str):
    out_dir = RUNS_DIR / run_id
    fig_dir = out_dir / "figures"
    if not fig_dir.exists():
        return render_template("error.html", title="Not found", message="No figures for this run."), 404
    return send_from_directory(fig_dir, name)


if __name__ == "__main__":
    # For a student demo, the built-in dev server is OK.
    # Run: python app.py  -> open http://127.0.0.1:5000
    debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug_enabled, threaded=True)
