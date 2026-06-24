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
  - both are incorrect but for different reasons (incorrectness-based)
  - one is invalid (empty/error) and the other is not (invalidity-based)

Demographic inconsistency (this prototype)
-------------------------------------------
A demographic-renamed variant is demographically inconsistent when:
  - it disagrees with the original, AND
  - the neutral-renamed variant agrees with the original
This isolates demographic signal from mere name-plausibility loss.

Self-consistency filter (mitigation, from MUCOCO Appendix I + Task 2 proposal)
-------------------------------------------------------------------------------
Trust an answer only when original and neutral-renamed versions agree.
Abstain otherwise. Trades coverage for precision.
"""


def normalize(value: str) -> str:
    """
    Normalize a predicted or expected value for comparison.
    Tries eval() to canonicalize list/bool/int representations,
    falls back to lowercased string.
    """
    v = str(value).strip()
    try:
        return str(eval(v))
    except Exception:
        return v.lower()


def is_correct(answer: str, expected: str) -> bool:
    """True if the LLM answer matches the expected output after normalization."""
    return normalize(answer) == normalize(expected)


def is_invalid(answer: str) -> bool:
    """True if the answer is an API error or empty."""
    a = str(answer).strip()
    return not a or a.upper().startswith("ERROR") or a.upper().startswith("API_ERROR")


def consistency_verdict(ans_orig: str, ans_mut: str, exp: str) -> dict:
    """
    Classify a (original, mutated) answer pair.

    Returns a dict with:
      inconsistent: bool
      type: str  ('consistent' | 'correctness' | 'incorrectness' | 'invalidity')
      orig_correct: bool
      mut_correct: bool
    """
    orig_invalid = is_invalid(ans_orig)
    mut_invalid = is_invalid(ans_mut)
    orig_correct = is_correct(ans_orig, exp) if not orig_invalid else False
    mut_correct = is_correct(ans_mut, exp) if not mut_invalid else False

    if orig_invalid or mut_invalid:
        inconsistent = orig_invalid != mut_invalid
        return {
            "inconsistent": inconsistent,
            "type": "invalidity" if inconsistent else "consistent",
            "orig_correct": orig_correct,
            "mut_correct": mut_correct,
        }

    if orig_correct == mut_correct:
        return {
            "inconsistent": False,
            "type": "consistent",
            "orig_correct": orig_correct,
            "mut_correct": mut_correct,
        }

    # one correct, one not
    return {
        "inconsistent": True,
        "type": "correctness",
        "orig_correct": orig_correct,
        "mut_correct": mut_correct,
    }


def is_demographically_inconsistent(
    ans_orig: str,
    ans_neutral: str,
    ans_demo: str,
    exp: str,
) -> bool:
    """
    True if demographic inconsistency is detected.

    Definition: the demographic variant disagrees with the original,
    while the neutral variant agrees with the original.
    This attribution isolates demographic signal from general
    name-plausibility effects.
    """
    orig_vs_neutral = consistency_verdict(ans_orig, ans_neutral, exp)
    orig_vs_demo = consistency_verdict(ans_orig, ans_demo, exp)
    return (
        orig_vs_demo["inconsistent"]
        and not orig_vs_neutral["inconsistent"]
    )


# ---------------------------------------------------------------------------
# Self-consistency filter (mitigation)
# ---------------------------------------------------------------------------

def self_consistency_filter(ans_orig: str, ans_neutral: str, exp: str) -> dict:
    """
    Lightweight mitigation from MUCOCO Appendix I + Task 2 proposal.

    If original and neutral-renamed answers agree -> retain the answer.
    If they disagree -> abstain (flag for human review).

    Returns:
      retained: bool
      answer:   str (ans_orig if retained, else 'ABSTAIN')
      correct:  bool | None (None if abstained)
    """
    if normalize(ans_orig) == normalize(ans_neutral):
        return {
            "retained": True,
            "answer": ans_orig,
            "correct": is_correct(ans_orig, exp),
        }
    return {
        "retained": False,
        "answer": "ABSTAIN",
        "correct": None,
    }
