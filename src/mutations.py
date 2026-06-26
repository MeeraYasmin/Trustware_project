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

Bug fixed (v2)
--------------
The original _safe_rename used str.replace(), which replaces ALL occurrences
of the substring -- including when the param name appears inside a Python
keyword or another identifier.

Example: param "n", renamed to "value"
  BAD (str.replace):  "return n % 2"  ->  "returvalue value % 2"
  BAD (str.replace):  "range(1, n+1)" ->  "ravamluege(1, value+1)"

Fix: _token_replace uses re.sub with word-boundary anchors (\b) so only
complete identifier tokens are replaced, never substrings of other words.
"""

import re
import random
import string

# ---------------------------------------------------------------------------
# Demographic name sets
# ---------------------------------------------------------------------------

DEMOGRAPHIC_GROUPS = {
    "neutral": {
        "fn":    "compute",
        "v1":    "value",
        "v2":    "data",
        "label": "neutral rename",
    },
    "male_western": {
        "fn":    "john_calc",
        "v1":    "michael",
        "v2":    "william",
        "label": "male / western",
    },
    "female_western": {
        "fn":    "emily_calc",
        "v1":    "jessica",
        "v2":    "sarah",
        "label": "female / western",
    },
    "male_south_asian": {
        "fn":    "raj_calc",
        "v1":    "vikram",
        "v2":    "arjun",
        "label": "male / south asian",
    },
    "female_south_asian": {
        "fn":    "priya_calc",
        "v1":    "ananya",
        "v2":    "deepa",
        "label": "female / south asian",
    },
}

ALL_MUTATION_NAMES = ["original", "random"] + [f"demo_{g}" for g in DEMOGRAPHIC_GROUPS]


# ---------------------------------------------------------------------------
# Core rename primitive (word-boundary safe)
# ---------------------------------------------------------------------------

def _token_replace(text: str, old: str, new: str) -> str:
    """
    Replace all whole-token occurrences of `old` with `new`.

    Uses regex word boundaries (\b) so that renaming param "n" to "value"
    does NOT corrupt "return" -> "returvalue" or "range" -> "ravalue ge".
    This is the core fix over the original str.replace() implementation.
    """
    return re.sub(r'\b' + re.escape(old) + r'\b', new, text)


def _safe_rename(code: str, fn_new: str, v1_new: str, v2_new: str) -> str:
    """
    Rename function name and up to first two parameters using
    word-boundary-safe token replacement.

    Steps:
    1. Parse the 'def' line to extract the old function name and param names.
    2. Walk every line of code applying _token_replace for each old->new
       mapping (function name first, then params left-to-right).

    The AST of the program is structurally identical before and after --
    only identifier tokens change, so the program's semantics are preserved
    and HInter's dependency invariant passes trivially.
    """
    lines  = code.split('\n')
    fn_old: str | None = None
    p1_old: str | None = None
    p2_old: str | None = None

    for line in lines:
        if fn_old is None and line.strip().startswith('def '):
            after_def  = line.strip()[4:]
            fn_old     = after_def.split('(')[0].strip()
            params_raw = after_def.split('(')[1].split(')')[0]
            params     = [p.strip() for p in params_raw.split(',') if p.strip()]
            p1_old     = params[0] if len(params) > 0 else None
            p2_old     = params[1] if len(params) > 1 else None
            break

    out = []
    for line in lines:
        nl = line
        if fn_old:
            nl = _token_replace(nl, fn_old, fn_new)
        if p1_old and v1_new:
            nl = _token_replace(nl, p1_old, v1_new)
        if p2_old and v2_new:
            nl = _token_replace(nl, p2_old, v2_new)
        out.append(nl)

    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Public mutation operators
# ---------------------------------------------------------------------------

def random_rename(code: str, seed: int = 42) -> str:
    """
    MUCOCO-style random lexical mutation.

    Replaces function name and first two parameters with random lowercase
    strings (seeded for reproducibility). Replicates MUCOCO's 'random'
    mutation operator, which produces the highest single-operator
    inconsistency rate (16.28%, Table 4).
    """
    rng   = random.Random(seed)
    chars = string.ascii_lowercase
    fn    = ''.join(rng.choices(chars, k=9))
    v1    = ''.join(rng.choices(chars, k=6))
    v2    = ''.join(rng.choices(chars, k=6))
    return _safe_rename(code, fn, v1, v2)


def neutral_rename(code: str) -> str:
    """
    Control condition: plausible but demographically neutral names.

    Removes meaningful names (like random_rename does) but replaces them
    with semantically plausible generic terms ('compute', 'value', 'data').

    Methodological role: any inconsistency gap between this condition and
    a demographic condition is attributable to demographic signal, not to
    name plausibility loss -- because both conditions replace the original
    meaningful name with an equally unfamiliar one.
    """
    g = DEMOGRAPHIC_GROUPS['neutral']
    return _safe_rename(code, g['fn'], g['v1'], g['v2'])


def demographic_rename(code: str, group: str) -> str:
    """
    HInter-inspired demographic lexical mutation.

    Replaces function/parameter names with demographically loaded equivalents
    (e.g. 'john_calc' / 'michael' for male_western). Names were chosen to
    carry clear demographic signal in Western software engineering culture,
    analogous to the demographic name substitutions used in resume-audit
    studies (Bertrand & Mullainathan).

    Args:
        code:  original Python function source
        group: key in DEMOGRAPHIC_GROUPS (e.g. 'male_western')

    Returns:
        Source with identifier names replaced by demographically loaded ones.

    Raises:
        ValueError if group is not in DEMOGRAPHIC_GROUPS.
    """
    if group not in DEMOGRAPHIC_GROUPS:
        raise ValueError(
            f"Unknown demographic group: {group!r}. "
            f"Choose from: {list(DEMOGRAPHIC_GROUPS)}"
        )
    g = DEMOGRAPHIC_GROUPS[group]
    return _safe_rename(code, g['fn'], g['v1'], g['v2'])


def apply_all_mutations(code: str) -> dict:
    """
    Apply every mutation operator and return a dict of
    {mutation_name: mutated_code} for all operators.
    """
    result = {'original': code, 'random': random_rename(code)}
    for grp in DEMOGRAPHIC_GROUPS:
        result[f'demo_{grp}'] = demographic_rename(code, grp)
    return result


def mutation_label(mut_name: str) -> str:
    """Human-readable label for a mutation name."""
    if mut_name == 'original':
        return 'original'
    if mut_name == 'random':
        return 'random rename'
    grp = mut_name.replace('demo_', '')
    return DEMOGRAPHIC_GROUPS.get(grp, {}).get('label', mut_name)
