"""
Program bank: loaded at runtime from CruxEval (Gu et al., ICML 2024).

    Primary source:  HuggingFace Hub  cruxeval-org/cruxeval  (test split, 800 samples)
    Fallback source: GitHub raw JSONL https://raw.githubusercontent.com/facebookresearch/cruxeval/main/data/cruxeval.jsonl

The module tries HuggingFace first. If that fails (no `datasets` package,
or no network access), it falls back to fetching the JSONL directly from
GitHub using only Python stdlib (urllib). No separate fetch_cruxeval.py
script is needed -- everything lives here.

PROGRAMS is a module-level list populated once at import time.
Each entry is a dict with keys:
    id    -- CruxEval sample ID, e.g. "sample_178"
    desc  -- human-readable label including param names
    code  -- Python function source (original name is always "f")
    inp   -- input string, e.g. "[1, 2, 3], 4"
    exp   -- expected output string, e.g. "[2, 3]"

Selection criteria (applied to all 800 samples)
------------------------------------------------
1.  Syntactically valid Python.
2.  Exactly one FunctionDef, no Import/ImportFrom nodes
    (mutation must be self-contained).
3.  At least one parameter (renaming needs something to rename).
4.  At most 20 lines (keeps LLM prompts manageable).
5.  Output is not None/empty, not a set/dict literal (ordering is
    non-deterministic), not an error traceback.

Stratified sample (seed = 2026)
--------------------------------
    25  single-letter params  (e.g. n, s, a) -- highest mutation risk
    10  two-letter params
    15  three-or-more-letter params
    ---
    50  programs total  (always >= 20 guaranteed)

CLI
---
    python src/programs.py          # prints first 5 programs as a sanity check
    python src/programs.py --all    # prints all 50
"""

import ast
import json
import random
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_JSONL_URL = (
    "https://raw.githubusercontent.com/facebookresearch/cruxeval"
    "/main/data/cruxeval.jsonl"
)

SEED     = 2026
N_SHORT  = 25   # single-letter params
N_MEDIUM = 10   # two-letter params
N_LONGER = 15   # three-or-more-letter params
MIN_PROGRAMS = 20


# ---------------------------------------------------------------------------
# Shared eligibility helpers (used by both loaders)
# ---------------------------------------------------------------------------

def _eligible_params(code: str) -> Optional[list[str]]:
    """
    Return the parameter name list if the program is eligible, else None.

    Rules:
        - Parses as valid Python
        - Exactly one FunctionDef, zero imports
        - At least one parameter
        - At most 20 lines
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    nodes    = list(ast.walk(tree))
    funcdefs = [n for n in nodes if isinstance(n, ast.FunctionDef)]
    imports  = [n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))]

    if len(funcdefs) != 1 or imports:
        return None
    if not funcdefs[0].args.args:
        return None
    if len(code.splitlines()) > 20:
        return None

    return [a.arg for a in funcdefs[0].args.args]


def _is_eligible_output(output: str) -> bool:
    """
    True if the expected output is suitable for string-match comparison.

    Excludes: None/empty, set/dict literals (unstable ordering),
    error tracebacks.
    """
    out = output.strip()
    if not out or out in ("None", "none"):
        return False
    if out.startswith("{"):
        return False
    if any(x in out for x in ("Error", "Exception", "Traceback")):
        return False
    return True


def _build_entry(raw_id: str, code: str, inp: str, output: str,
                 params: list[str]) -> dict:
    """Normalise a raw CruxEval row into the schema experiment.py expects."""
    return {
        "id":   raw_id,
        "desc": f"CruxEval {raw_id} | params: {params}",
        "code": code,
        "inp":  inp,
        "exp":  output,
    }


def _stratified_sample(eligible: list[dict],
                       n_short: int, n_medium: int, n_longer: int,
                       seed: int) -> list[dict]:
    """
    Split eligible programs by minimum param length and sample from each stratum.
    Each dict must have '_min_param_len' set before calling this.
    """
    short_p  = [s for s in eligible if s["_min_param_len"] == 1]
    medium_p = [s for s in eligible if s["_min_param_len"] == 2]
    longer_p = [s for s in eligible if s["_min_param_len"] >= 3]

    rng = random.Random(seed)
    selected = (
        rng.sample(short_p,  min(n_short,  len(short_p)))  +
        rng.sample(medium_p, min(n_medium, len(medium_p))) +
        rng.sample(longer_p, min(n_longer, len(longer_p)))
    )
    rng.shuffle(selected)

    print(
        f"  Selected {len(selected)} programs "
        f"({min(n_short,  len(short_p))} single-letter, "
        f"{min(n_medium, len(medium_p))} two-letter, "
        f"{min(n_longer, len(longer_p))} longer params)."
    )
    return selected


# ---------------------------------------------------------------------------
# Loader A: HuggingFace Hub (primary)
# ---------------------------------------------------------------------------

def _load_from_huggingface(n_short: int, n_medium: int, n_longer: int,
                            seed: int) -> list[dict]:
    """
    Load CruxEval via the `datasets` library from HuggingFace Hub.
    Raises ImportError if `datasets` is not installed.
    Raises any network/hub error as-is so the caller can fall back.
    """
    from datasets import load_dataset  # may raise ImportError

    print("Loading CruxEval from HuggingFace Hub (cruxeval-org/cruxeval) ...")
    raw = load_dataset("cruxeval-org/cruxeval", split="test")
    print(f"  Downloaded {len(raw)} samples.")

    eligible = []
    for row in raw:
        if not _is_eligible_output(row["output"]):
            continue
        params = _eligible_params(row["code"])
        if params is None:
            continue
        eligible.append({
            **_build_entry(row["id"], row["code"], row["input"], row["output"], params),
            "_min_param_len": min(len(p) for p in params),
        })

    print(f"  Eligible after filtering: {len(eligible)} / {len(raw)}")
    selected = _stratified_sample(eligible, n_short, n_medium, n_longer, seed)
    # strip internal metadata before returning
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in selected]


# ---------------------------------------------------------------------------
# Loader B: GitHub JSONL (fallback, stdlib only)
# ---------------------------------------------------------------------------

def _load_from_github(n_short: int, n_medium: int, n_longer: int,
                       seed: int) -> list[dict]:
    """
    Fetch the CruxEval JSONL directly from GitHub using urllib (no extra deps).
    Used as fallback when the `datasets` package is unavailable or the Hub
    is unreachable.
    """
    print(f"Falling back: fetching CruxEval JSONL from GitHub ...")
    print(f"  URL: {GITHUB_JSONL_URL}")

    with urllib.request.urlopen(GITHUB_JSONL_URL, timeout=30) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    samples = [json.loads(line) for line in lines if line.strip()]
    print(f"  Downloaded {len(samples)} samples.")

    eligible = []
    for s in samples:
        if not _is_eligible_output(s["output"]):
            continue
        params = _eligible_params(s["code"])
        if params is None:
            continue
        eligible.append({
            **_build_entry(s["id"], s["code"], s["input"], s["output"], params),
            "_min_param_len": min(len(p) for p in params),
        })

    print(f"  Eligible after filtering: {len(eligible)} / {len(samples)}")
    selected = _stratified_sample(eligible, n_short, n_medium, n_longer, seed)
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in selected]


# ---------------------------------------------------------------------------
# Public loader (tries HuggingFace, falls back to GitHub)
# ---------------------------------------------------------------------------

def load_programs(
    n_short:      int = N_SHORT,
    n_medium:     int = N_MEDIUM,
    n_longer:     int = N_LONGER,
    seed:         int = SEED,
    min_programs: int = MIN_PROGRAMS,
) -> list[dict]:
    """
    Load a stratified sample of CruxEval programs.

    Tries HuggingFace Hub first; falls back to GitHub JSONL if the
    `datasets` package is missing or the Hub is unreachable.

    Args:
        n_short:      programs with single-letter params  (default 25).
        n_medium:     programs with two-letter params     (default 10).
        n_longer:     programs with 3+-letter params      (default 15).
        seed:         random seed for reproducibility     (default 2026).
        min_programs: raise RuntimeError if fewer programs pass filters.

    Returns:
        List of dicts with keys: id, desc, code, inp, exp.
    """
    programs = None
    errors   = []

    # --- try HuggingFace first ---
    try:
        programs = _load_from_huggingface(n_short, n_medium, n_longer, seed)
    except ImportError:
        errors.append("  HuggingFace: `datasets` package not installed.")
    except Exception as exc:
        errors.append(f"  HuggingFace: {exc}")

    # --- fall back to GitHub ---
    if programs is None:
        for msg in errors:
            print(msg)
        try:
            programs = _load_from_github(n_short, n_medium, n_longer, seed)
        except Exception as exc:
            raise RuntimeError(
                f"Both loaders failed.\n"
                f"HuggingFace errors: {errors}\n"
                f"GitHub error: {exc}\n"
                "Check your internet connection or install `datasets`."
            ) from exc

    if len(programs) < min_programs:
        raise RuntimeError(
            f"Only {len(programs)} programs loaded "
            f"(min_programs={min_programs} required)."
        )

    print(f"  programs.py ready: {len(programs)} programs (seed={seed})\n")
    return programs


# ---------------------------------------------------------------------------
# Module-level PROGRAMS list -- populated once at import time
# ---------------------------------------------------------------------------

PROGRAMS: list[dict] = load_programs()


# ---------------------------------------------------------------------------
# CLI: python src/programs.py  [--all]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    show_all = "--all" in sys.argv
    items    = PROGRAMS if show_all else PROGRAMS[:5]

    print(f"\nLoaded {len(PROGRAMS)} programs total.\n")
    for i, p in enumerate(items):
        print(f"[{i:>2}] {p['id']}")
        print(f"     inp : {p['inp']}")
        print(f"     exp : {p['exp']}")
        print(f"     code:")
        for line in p["code"].splitlines():
            print(f"         {line}")
        print()
