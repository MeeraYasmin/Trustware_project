"""
Mutation operators for demographic lexical mutation testing.

Design rationale
----------------
MUCOCO (Chua et al., 2026) establishes that lexical mutations produce the
highest inconsistency rate (16.28%) of all mutation types, and attributes
this to LLM reliance on meaningful variable names rather than program
semantics. HInter (Souani et al., 2025) establishes that demographic
attribute substitutions expose bias invisible to single-attribute testing.

This module implements:
  1. random_rename   -- MUCOCO's baseline lexical mutation
  2. neutral_rename  -- control condition: plausible but demographically
                        neutral names, isolates plausibility from
                        demographic signal
  3. demographic_*   -- HInter-inspired: demographically loaded names
                        covering gender x ethnicity/origin

The neutral_rename condition is the key methodological contribution of
this prototype: any inconsistency above the neutral baseline that appears
specifically in one demographic group is attributable to demographic
signal rather than loss of meaningful names.
"""

import random
import string

# ---------------------------------------------------------------------------
# Demographic name sets
# Each group specifies a function name and two parameter name replacements.
# Names chosen to carry clear demographic signal in Western CS culture
# (similar to the "resume audit" literature, e.g. Bertrand & Mullainathan).
# ---------------------------------------------------------------------------

DEMOGRAPHIC_GROUPS = {
    "neutral": {
        "fn": "compute",
        "v1": "value",
        "v2": "data",
        "label": "neutral rename",
    },
    "male_western": {
        "fn": "john_calc",
        "v1": "michael",
        "v2": "william",
        "label": "male / western",
    },
    "female_western": {
        "fn": "emily_calc",
        "v1": "jessica",
        "v2": "sarah",
        "label": "female / western",
    },
    "male_south_asian": {
        "fn": "raj_calc",
        "v1": "vikram",
        "v2": "arjun",
        "label": "male / south asian",
    },
    "female_south_asian": {
        "fn": "priya_calc",
        "v1": "ananya",
        "v2": "deepa",
        "label": "female / south asian",
    },
}

ALL_MUTATION_NAMES = ["original", "random"] + [f"demo_{g}" for g in DEMOGRAPHIC_GROUPS]


def _safe_rename(code: str, fn_new: str, v1_new: str, v2_new: str) -> str:
    """
    Rename function name and up to first two parameters.

    Uses string token replacement (not AST) for simplicity; safe here
    because we replace short, specific identifiers that appear as exact
    substrings. The AST of the program is unchanged -- the dependency
    invariant HInter uses would pass trivially for pure lexical mutations,
    which is a strength: no syntactic distortion, only name-token change.
    """
    lines = code.split("\n")
    fn_old = None
    p1_old = None
    p2_old = None

    for i, line in enumerate(lines):
        if fn_old is None and line.strip().startswith("def "):
            after_def = line.strip()[4:]
            fn_old = after_def.split("(")[0].strip()
            params_raw = after_def.split("(")[1].split(")")[0]
            params = [p.strip() for p in params_raw.split(",") if p.strip()]
            p1_old = params[0] if len(params) > 0 else None
            p2_old = params[1] if len(params) > 1 else None

    out = []
    for line in lines:
        nl = line
        if fn_old:
            nl = nl.replace(fn_old, fn_new)
        if p1_old and v1_new:
            nl = nl.replace(p1_old, v1_new)
        if p2_old and v2_new:
            nl = nl.replace(p2_old, v2_new)
        out.append(nl)
    return "\n".join(out)


def random_rename(code: str, seed: int = 42) -> str:
    """
    MUCOCO-style random lexical mutation.

    Replaces function name and first two parameters with random lowercase
    strings. Replicates MUCOCO's 'random' mutation operator, which
    produces the highest single-operator inconsistency rate (Table 4).
    """
    rng = random.Random(seed)
    chars = string.ascii_lowercase
    fn = "".join(rng.choices(chars, k=9))
    v1 = "".join(rng.choices(chars, k=6))
    v2 = "".join(rng.choices(chars, k=6))
    return _safe_rename(code, fn, v1, v2)


def neutral_rename(code: str) -> str:
    """
    Control condition: plausible but demographically neutral names.

    Removes meaningful names like random_rename does, but replaces them
    with semantically plausible generic terms ('compute', 'value', 'data').
    Any inconsistency gap between this condition and demographic conditions
    is attributable to demographic signal, not to name plausibility loss.
    """
    g = DEMOGRAPHIC_GROUPS["neutral"]
    return _safe_rename(code, g["fn"], g["v1"], g["v2"])


def demographic_rename(code: str, group: str) -> str:
    """
    HInter-inspired demographic lexical mutation.

    Args:
        code:  original Python function source
        group: key in DEMOGRAPHIC_GROUPS (e.g. 'male_western')

    Returns:
        Source with function/parameter names replaced by demographically
        loaded equivalents.
    """
    if group not in DEMOGRAPHIC_GROUPS:
        raise ValueError(f"Unknown demographic group: {group!r}. "
                         f"Choose from: {list(DEMOGRAPHIC_GROUPS)}")
    g = DEMOGRAPHIC_GROUPS[group]
    return _safe_rename(code, g["fn"], g["v1"], g["v2"])


def apply_all_mutations(code: str) -> dict:
    """
    Return a dict of {mutation_name: mutated_code} for all operators.
    """
    results = {"original": code, "random": random_rename(code)}
    for grp in DEMOGRAPHIC_GROUPS:
        results[f"demo_{grp}"] = demographic_rename(code, grp)
    return results


def mutation_label(mut_name: str) -> str:
    """Human-readable label for a mutation name."""
    if mut_name == "original":
        return "original"
    if mut_name == "random":
        return "random rename"
    grp = mut_name.replace("demo_", "")
    return DEMOGRAPHIC_GROUPS.get(grp, {}).get("label", mut_name)
