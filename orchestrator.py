"""
Autoresearch Orchestrator — batch-mode autonomous experiment runner.

Designed to run as a **nightly batch job**, NOT a continuous daemon.
Each invocation runs a bounded number of experiments (default 12 ≈ 1 hour),
persists all state to ``state.json``, then exits cleanly.

State file (``state.json``) tracks:
    - Current branch / tag
    - Cumulative experiment counter (across nightly runs)
    - Consecutive failure counter (survives restarts)
    - Best val_bpb achieved
    - Timestamps of each nightly run

The next invocation resumes exactly where the previous one left off.

Usage:
    # Nightly cron / systemd timer (inside crsai-pytorch container):
    AUTORESEARCH_PROFILE=rtx5060 python3 orchestrator.py --tag apr

    # Or from host:
    docker exec -w /workspace/autoresearch crsai-pytorch \\
        env AUTORESEARCH_PROFILE=rtx5060 python3 orchestrator.py --tag apr

    # Override batch size (default 12 experiments ≈ 1 hour):
    python3 orchestrator.py --tag apr --batch-size 6
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_URL = os.getenv("LLM_URL", "http://crsai-vllm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-cascade-2-nvfp4")
MAX_EXPERIMENT_SECONDS = 600  # kill if > 10 min
TRAIN_CMD = "python3 train.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"
STATE_FILE = "state.json"
DEFAULT_BATCH_SIZE = 12  # ~1 hour at 5 min/experiment
MAX_CONSECUTIVE_FAILURES = 3
RAG_RESERVATION_SCRIPT = os.getenv(
    "AUTORESEARCH_RAG_RESERVATION_SCRIPT",
    "/workspace/scripts/rag_gpu_reservation.sh",
)

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


def _empty_state() -> dict:
    """Return a fresh state dict with defaults."""
    return {
        "tag": None,
        "branch": None,
        "total_experiments": 0,
        "consecutive_failures": 0,
        "best_val_bpb": None,
        "runs": [],  # list of {date, batch_size, experiments_run, best_bpb_after}
    }


def load_state() -> dict:
    """Load persisted state from STATE_FILE, or return defaults."""
    path = Path(STATE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return _empty_state()


def save_state(state: dict) -> None:
    """Atomically persist state to STATE_FILE."""
    tmp = Path(STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def log(msg: str) -> None:
    """Timestamped log line."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def reserve_gpu_for_ml() -> None:
    """Force MCP-RAG to CPU before starting a GPU-bound AutoResearch run."""
    proc = subprocess.run(
        ["bash", RAG_RESERVATION_SCRIPT, "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if output:
        log(output[-500:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"GPU reservation failed via {RAG_RESERVATION_SCRIPT}: {output[-2000:]}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: str, timeout: int | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a shell command, return CompletedProcess."""
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=cwd,
    )


def git_short_hash() -> str:
    r = run("git rev-parse --short HEAD")
    return r.stdout.strip()


def git_commit(msg: str) -> str:
    run("git add train.py")
    run(f'git commit -m "{msg}"')
    return git_short_hash()


def git_reset_hard(ref: str = "HEAD~1") -> None:
    run(f"git reset --hard {ref}")


def read_train_py() -> str:
    return Path("train.py").read_text()


def write_train_py(content: str) -> None:
    Path("train.py").write_text(content)


def init_results_tsv() -> None:
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def append_result(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    line = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)


def read_results() -> str:
    if Path(RESULTS_FILE).exists():
        return Path(RESULTS_FILE).read_text()
    return ""


def parse_run_log(log_path: str = RUN_LOG) -> dict:
    """Extract val_bpb and peak_vram_mb from run.log."""
    result = {"val_bpb": None, "peak_vram_mb": None, "crashed": False}
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        result["crashed"] = True
        return result

    # Check for FAIL or crash
    if "FAIL" in text or "Error" in text or "Traceback" in text:
        result["crashed"] = True

    for line in text.splitlines():
        if line.startswith("val_bpb:"):
            result["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            result["peak_vram_mb"] = float(line.split(":")[1].strip())

    if result["val_bpb"] is None:
        result["crashed"] = True

    return result


def best_val_bpb() -> float | None:
    """Return best val_bpb from results.tsv (keeps only)."""
    results = read_results()
    best = None
    for line in results.strip().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            bpb = float(parts[1])
            if best is None or bpb < best:
                best = bpb
    return best


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def check_vllm_ready(retries: int = 3, delay: int = 10) -> bool:
    """Verify vLLM is reachable before starting the experiment loop."""
    import requests

    url = f"{LLM_URL}/models"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            log(f"vLLM health check passed (attempt {attempt})")
            return True
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            log(f"vLLM not ready (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(delay)
    return False


def call_llm(system: str, user: str, max_tokens: int = 8192) -> str:
    """Call local vLLM endpoint. Uses requests (already installed)."""
    import requests

    resp = requests.post(
        f"{LLM_URL}/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


SYSTEM_PROMPT = """You are an expert ML researcher running autonomous training experiments.
You modify train.py to improve val_bpb (validation bits per byte — lower is better).
The training runs for exactly 5 minutes. You can change anything in train.py:
architecture, optimizer, hyperparameters, batch size, model size, etc.

CONSTRAINTS:
- Do NOT modify prepare.py or add packages
- Keep changes focused — one idea per experiment
- Simpler is better at equal performance
- VRAM is limited (~5-8 GB available)
- The AUTORESEARCH_PROFILE override block near line 440 MUST be preserved

RESPONSE FORMAT:
Return ONLY a JSON object with these keys:
{
  "description": "brief description of what this experiment tries",
  "replacements": [
    {"old": "exact lines to find in train.py", "new": "replacement lines"}
  ]
}

Each replacement is a search-and-replace pair. Use the EXACT text from train.py
including whitespace. Keep replacements minimal — only the lines that change.
Do NOT include markdown code fences. Return raw JSON only."""


def apply_replacements(train_py: str, replacements: list[dict]) -> str:
    """Apply a list of {old, new} replacements to train.py content."""
    result = train_py
    for r in replacements:
        old = r.get("old", "")
        new = r.get("new", "")
        if old not in result:
            raise ValueError(f"Replacement target not found in train.py: {old[:80]!r}")
        result = result.replace(old, new, 1)
    return result


def propose_experiment(train_py: str, results_history: str) -> dict:
    """Ask the LLM to propose a train.py modification."""
    user_msg = f"""Current train.py:
```python
{train_py}
```

Experiment history (TSV):
```
{results_history}
```

Current best val_bpb: {best_val_bpb() or 'no results yet — this is the first run'}

Propose a single focused modification to improve val_bpb.
Return search-and-replace pairs (exact text from the file) and a description."""

    response = call_llm(SYSTEM_PROMPT, user_msg, max_tokens=8192)

    # Strip thinking tags (reasoning model outputs <think>...</think> before JSON)
    response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    # If <think> started but never closed, strip everything before the first {
    if "<think>" in response:
        idx = response.find("{")
        if idx >= 0:
            response = response[idx:]

    # If response starts with non-JSON text (reasoning without tags), find first {
    stripped = response.lstrip()
    if stripped and stripped[0] != "{":
        idx = response.find("{")
        if idx >= 0:
            response = response[idx:]

    # Strip markdown fences if present
    response = re.sub(r"^```(?:json)?\n?", "", response.strip())
    response = re.sub(r"\n?```$", "", response.strip())

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse LLM response as JSON:\n{response[:500]}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Run train.py and return parsed results."""
    print(f"  Running training ({MAX_EXPERIMENT_SECONDS}s timeout)...", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.run(
                TRAIN_CMD.split(),
                stdout=log_file, stderr=subprocess.STDOUT,
                timeout=MAX_EXPERIMENT_SECONDS,
                env=env,
            )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT — experiment killed", flush=True)
        return {"val_bpb": None, "peak_vram_mb": None, "crashed": True}

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.0f}s", flush=True)

    return parse_run_log()


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch orchestrator (batch mode — run N experiments then exit)")
    parser.add_argument("--tag", required=True,
                        help="Experiment run tag (e.g. apr). Branch = autoresearch/<tag>")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Experiments to run this invocation (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the baseline, then exit")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load / initialise persistent state
    # ------------------------------------------------------------------
    state = load_state()
    branch = f"autoresearch/{args.tag}"
    run_start = datetime.now(timezone.utc).isoformat()

    # If tag changed, reset state for the new research track
    if state["tag"] != args.tag:
        log(f"New tag '{args.tag}' (previous: {state['tag']}). Resetting state.")
        state = _empty_state()
        state["tag"] = args.tag
        state["branch"] = branch

    log("=== Autoresearch Orchestrator (batch) ===")
    log(f"Branch: {branch}")
    log(f"LLM: {LLM_URL} / {LLM_MODEL}")
    log(f"Profile: {os.getenv('AUTORESEARCH_PROFILE', 'default')}")
    log(f"Batch size: {args.batch_size}")
    log(f"Cumulative experiments so far: {state['total_experiments']}")
    log(f"Consecutive failures carried over: {state['consecutive_failures']}")

    # ------------------------------------------------------------------
    # Reserve GPU 1 for AutoResearch before any training starts
    # ------------------------------------------------------------------
    try:
        reserve_gpu_for_ml()
    except Exception as exc:
        log(f"FATAL: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Preflight: ensure vLLM is reachable
    # ------------------------------------------------------------------
    if not check_vllm_ready():
        log("FATAL: vLLM is not reachable at " + LLM_URL)
        log("Start it first: docker ps --filter name=crsai-vllm")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Setup branch
    # ------------------------------------------------------------------
    r = run(f"git rev-parse --verify {branch}")
    if r.returncode != 0:
        run(f"git checkout -b {branch}")
        log(f"Created branch: {branch}")
    else:
        run(f"git checkout {branch}")
        log(f"Checked out existing branch: {branch}")

    init_results_tsv()

    # ------------------------------------------------------------------
    # Baseline (only needed on first-ever run)
    # ------------------------------------------------------------------
    if best_val_bpb() is None:
        log("--- Baseline Run ---")
        result = run_experiment()
        commit = git_short_hash()
        if result["crashed"]:
            log("BASELINE CRASHED — check run.log")
            append_result(commit, 0.0, 0.0, "crash", "baseline")
            state["consecutive_failures"] += 1
            save_state(state)
            sys.exit(1)

        bpb = result["val_bpb"]
        mem = result["peak_vram_mb"] / 1024 if result["peak_vram_mb"] else 0
        append_result(commit, bpb, mem, "keep", "baseline")
        state["best_val_bpb"] = bpb
        save_state(state)
        log(f"Baseline: val_bpb={bpb:.6f}, memory={mem:.1f}GB")

    if args.baseline_only:
        log("Baseline complete. Exiting.")
        save_state(state)
        return

    # ------------------------------------------------------------------
    # Bounded experiment loop
    # ------------------------------------------------------------------
    batch_experiments = 0

    for _ in range(args.batch_size):
        # Abort on too many consecutive failures (persisted across runs)
        if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            log(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures (across runs). "
                "Resetting failure counter for next nightly run.")
            state["consecutive_failures"] = 0
            save_state(state)
            break

        state["total_experiments"] += 1
        batch_experiments += 1
        exp_id = state["total_experiments"]

        current_best = best_val_bpb()
        log(f"--- Experiment {exp_id} [{batch_experiments}/{args.batch_size}] "
            f"(best: {current_best:.6f}) ---")

        train_py = read_train_py()
        results_history = read_results()

        # --- Propose ---
        try:
            log("Querying LLM for proposal...")
            proposal = propose_experiment(train_py, results_history)
            description = proposal.get("description", "unknown modification")
            replacements = proposal.get("replacements", [])

            if not replacements:
                log("LLM returned no replacements, skipping")
                state["consecutive_failures"] += 1
                save_state(state)
                continue

            log(f"Proposal: {description}")
            log(f"Replacements: {len(replacements)}")
        except Exception as e:
            log(f"LLM error: {e}")
            state["consecutive_failures"] += 1
            save_state(state)
            time.sleep(30)
            continue

        # --- Apply ---
        try:
            new_train_py = apply_replacements(train_py, replacements)
            write_train_py(new_train_py)
        except ValueError as e:
            log(f"Replacement failed: {e}")
            state["consecutive_failures"] += 1
            save_state(state)
            continue

        state["consecutive_failures"] = 0
        commit = git_commit(f"exp{exp_id}: {description[:60]}")

        # --- Train ---
        result = run_experiment()

        if result["crashed"]:
            append_result(commit, 0.0, 0.0, "crash", description)
            log("CRASHED — reverting")
            git_reset_hard()
            save_state(state)
            continue

        bpb = result["val_bpb"]
        mem = result["peak_vram_mb"] / 1024 if result["peak_vram_mb"] else 0

        if bpb < current_best:
            append_result(commit, bpb, mem, "keep", description)
            improvement = current_best - bpb
            state["best_val_bpb"] = bpb
            log(f"KEEP — val_bpb={bpb:.6f} (improved by {improvement:.6f})")
        else:
            append_result(commit, bpb, mem, "discard", description)
            regression = bpb - current_best
            log(f"DISCARD — val_bpb={bpb:.6f} (worse by {regression:.6f})")
            git_reset_hard()

        # Persist after every experiment so a crash mid-batch loses at most 1
        save_state(state)

    # ------------------------------------------------------------------
    # End-of-batch bookkeeping
    # ------------------------------------------------------------------
    run_record = {
        "date": run_start,
        "batch_size": args.batch_size,
        "experiments_run": batch_experiments,
        "best_bpb_after": best_val_bpb(),
    }
    state["runs"].append(run_record)
    save_state(state)

    log("")
    log("=== Nightly Batch Summary ===")
    log(f"Experiments this run: {batch_experiments}")
    log(f"Cumulative experiments: {state['total_experiments']}")
    log(f"Best val_bpb: {best_val_bpb():.6f}")
    log(f"State persisted to: {STATE_FILE}")
    log(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
