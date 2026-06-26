"""
Test oracles: correctness and consistency, adapted from MUCOCO and HInter.

Correctness oracle (MUCOCO §3.2)
---------------------------------
Compares LLM answer to ground-truth expected output.
For output-prediction tasks, this is a normalized string match.

Consistency oracle (MUCOCO §3.2)
----------------------------------
Two test cases are inconsistent when:
  - one is correct and the other is not (correctness-based inconsistency)
  - both are incorrect but give different wrong answers (incorrectness-based)
  - one is invalid (empty/error) and the other is not (invalidity-based)

Demographic inconsistency (this prototype)
-------------------------------------------
A demographic-renamed variant is demographically inconsistent when:
  - it disagrees with the original, AND
  - the neutral-renamed variant agrees with the original.
This isolates demographic signal from mere name-plausibility loss --
the key methodological contribution of this prototype over MUCOCO and HInter.

Self-consistency filter (mitigation)
--------------------------------------
From MUCOCO Appendix I + Task 2 proposal.
Trust an answer only when original and neutral-renamed versions agree.
Abstain otherwise. Trades recall for precision.
"""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(value: str) -> str:
    """
    Normalise a predicted or expected value for comparison.

    Tries eval() to canonicalise list/bool/int representations
    (e.g. '[1,2,3]' == '[1, 2, 3]', 'True' == 'True'),
    falls back to lowercased string on any eval failure.
    """
    v = str(value).strip()
    try:
        return str(eval(v))
    except Exception:
        return v.lower()


# ---------------------------------------------------------------------------
# Correctness oracle
# ---------------------------------------------------------------------------

def is_correct(answer: str, expected: str) -> bool:
    """True if the LLM answer matches the expected output after normalisation."""
    return normalize(answer) == normalize(expected)


def is_invalid(answer: str) -> bool:
    """True if the answer is an API error or empty string."""
    a = str(answer).strip()
    return not a or a.upper().startswith("ERROR") or a.upper().startswith("API_ERROR")


# ---------------------------------------------------------------------------
# Consistency oracle
# ---------------------------------------------------------------------------

def consistency_verdict(ans_orig: str, ans_mut: str, exp: str) -> dict:
    """
    Classify an (original, mutated) answer pair as consistent or inconsistent.

    Inconsistency types (from MUCOCO §3.2):
      'correctness'  -- one answer is correct, the other is not
      'invalidity'   -- one answer is an error/empty, the other is not
      'consistent'   -- both agree (whether correct or not)

    Note: incorrectness-based inconsistency (both wrong but differently)
    is captured implicitly -- if normalize(ans_orig) != normalize(ans_mut)
    and neither is correct, consistency_verdict returns 'consistent' because
    MUCOCO defines inconsistency relative to the correctness axis. To detect
    pure answer divergence regardless of correctness, compare normalize()
    outputs directly.

    Returns a dict with:
        inconsistent:  bool
        type:          str
        orig_correct:  bool
        mut_correct:   bool
    """
    orig_invalid = is_invalid(ans_orig)
    mut_invalid  = is_invalid(ans_mut)
    orig_correct = is_correct(ans_orig, exp) if not orig_invalid else False
    mut_correct  = is_correct(ans_mut,  exp) if not mut_invalid  else False

    if orig_invalid or mut_invalid:
        inconsistent = orig_invalid != mut_invalid
        return {
            "inconsistent": inconsistent,
            "type":         "invalidity" if inconsistent else "consistent",
            "orig_correct": orig_correct,
            "mut_correct":  mut_correct,
        }

    if orig_correct == mut_correct:
        return {
            "inconsistent": False,
            "type":         "consistent",
            "orig_correct": orig_correct,
            "mut_correct":  mut_correct,
        }

    # one correct, one not
    return {
        "inconsistent": True,
        "type":         "correctness",
        "orig_correct": orig_correct,
        "mut_correct":  mut_correct,
    }


# ---------------------------------------------------------------------------
# Demographic inconsistency oracle
# ---------------------------------------------------------------------------

def is_demographically_inconsistent(
    ans_orig:    str,
    ans_neutral: str,
    ans_demo:    str,
    exp:         str,
) -> bool:
    """
    True if demographic inconsistency is detected for a single program.

    Definition
    ----------
    The demographic variant is demographically inconsistent when:
      (a) it disagrees with the original  (orig_vs_demo is inconsistent), AND
      (b) the neutral variant agrees with the original (orig_vs_neutral is consistent).

    Rationale
    ---------
    Condition (b) rules out the alternative explanation that any name change
    (not just a demographic one) would have caused the inconsistency. If the
    neutral variant also disagrees, the inconsistency is due to general
    name-plausibility loss -- not demographic signal specifically. Only when
    neutral agrees but demographic disagrees can we attribute the difference
    to the demographic content of the names.

    This is the key attribution test that neither MUCOCO nor HInter implements.
    """
    orig_vs_neutral = consistency_verdict(ans_orig, ans_neutral, exp)
    orig_vs_demo    = consistency_verdict(ans_orig, ans_demo,    exp)
    return (
        orig_vs_demo["inconsistent"]
        and not orig_vs_neutral["inconsistent"]
    )


# ---------------------------------------------------------------------------
# Self-consistency filter (mitigation)
# ---------------------------------------------------------------------------

def self_consistency_filter(ans_orig: str, ans_neutral: str, exp: str) -> dict:
    """
    Lightweight inference-time mitigation (MUCOCO Appendix I + Task 2 proposal).

    Mechanism
    ---------
    Run the same program twice: once with original names, once with neutral
    names. If both runs agree on an answer, retain it -- the model is stable
    on this input. If they disagree, abstain and flag for human review.

    Trade-off
    ---------
    Reduces inconsistency rate at the cost of coverage: some correct answers
    will be abstained on when the two runs happen to disagree. This mirrors
    MUCOCO's confidence-threshold result (inconsistency 8.03% -> 1.89% at
    threshold 0.99, Appendix I) but requires no probability scores -- only
    two API calls and a string comparison.

    Returns a dict with:
        retained:  bool   -- True if the answer is trusted
        answer:    str    -- ans_orig if retained, 'ABSTAIN' otherwise
        correct:   bool | None  -- None if abstained
    """
    if normalize(ans_orig) == normalize(ans_neutral):
        return {
            "retained": True,
            "answer":   ans_orig,
            "correct":  is_correct(ans_orig, exp),
        }
    return {
        "retained": False,
        "answer":   "ABSTAIN",
        "correct":  None,
    }
