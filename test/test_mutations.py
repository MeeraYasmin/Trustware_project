"""
Tests for mutation operators and oracles.
Run with: python -m pytest tests/
"""
import sys
import ast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from programs import PROGRAMS
from mutations import (
    random_rename, neutral_rename, demographic_rename,
    apply_all_mutations, mutation_label,
    DEMOGRAPHIC_GROUPS, ALL_MUTATION_NAMES, _safe_rename,
)
from oracle import (
    normalize, is_correct, consistency_verdict,
    is_demographically_inconsistent, self_consistency_filter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_CODE = (
    "def sum_list(nums):\n"
    "    total = 0\n"
    "    for x in nums:\n"
    "        total += x\n"
    "    return total"
)

# Programs with single-letter params that previously broke with str.replace()
P_CHECK_EVEN = "def check_even(n):\n    return n % 2 == 0"
P_FACTORIAL  = (
    "def factorial(n):\n"
    "    result = 1\n"
    "    for i in range(1, n + 1):\n"
    "        result *= i\n"
    "    return result"
)
P_SINGLE_S   = "def reverse_string(s):\n    return s[::-1]"


# ---------------------------------------------------------------------------
# Word-boundary safety (core regression suite for the str.replace() bug fix)
# ---------------------------------------------------------------------------

class TestWordBoundarySafety:
    """All cases broke with str.replace(); must pass with re.sub word-boundary."""

    def test_n_param_does_not_corrupt_return(self):
        result = demographic_rename(P_CHECK_EVEN, "neutral")
        assert "returvalue" not in result
        assert "return value" in result

    def test_n_param_does_not_corrupt_range(self):
        result = demographic_rename(P_FACTORIAL, "male_south_asian")
        assert "ravikramge" not in result
        assert "range(1, vikram + 1)" in result

    def test_n_param_does_not_corrupt_in_keyword(self):
        result = demographic_rename(P_FACTORIAL, "female_south_asian")
        assert "iananya" not in result
        assert " in range" in result

    def test_n_param_does_not_corrupt_new_fn_name(self):
        # john_calc contains 'n' -- must not become 'johvalue_calc'
        result = demographic_rename(P_CHECK_EVEN, "male_western")
        assert "def john_calc(michael)" in result
        assert "johvalue" not in result

    def test_random_rename_n_param_no_corruption(self):
        result = random_rename(P_CHECK_EVEN, seed=99)
        ast.parse(result)   # must be valid Python

    def test_random_rename_factorial_no_corruption(self):
        result = random_rename(P_FACTORIAL, seed=42)
        ast.parse(result)   # the old bug produced SyntaxError here

    def test_s_param_does_not_corrupt_seen(self):
        code = (
            "def remove_dups(s, lst):\n"
            "    seen = []\n"
            "    for item in lst:\n"
            "        if item not in seen:\n"
            "            seen.append(item)\n"
            "    return seen"
        )
        result = _safe_rename(code, "compute", "value", "data")
        assert "seen" in result
        assert "sevalue" not in result


# ---------------------------------------------------------------------------
# All CruxEval programs mutate to valid Python
# ---------------------------------------------------------------------------

class TestAllCruxEvalMutationsParseCleanly:
    """50 programs × 7 mutations = 350 variants must all be valid Python."""

    def test_all_mutations_parse(self):
        errors = []
        for p in PROGRAMS:
            muts = apply_all_mutations(p["code"])
            for mut_name, code in muts.items():
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors.append(f"{p['id']}/{mut_name}: {e}")
        assert errors == [], "Syntax errors found:\n" + "\n".join(errors)

    def test_program_count(self):
        assert len(PROGRAMS) >= 20, f"Only {len(PROGRAMS)} programs loaded"

    def test_all_mutation_names_present(self):
        for p in PROGRAMS[:3]:
            muts = apply_all_mutations(p["code"])
            for name in ALL_MUTATION_NAMES:
                assert name in muts, f"Missing mutation {name} for {p['id']}"

    def test_programs_have_required_keys(self):
        for p in PROGRAMS:
            for key in ("id", "desc", "code", "inp", "exp"):
                assert key in p, f"Missing key '{key}' in program {p.get('id')}"


# ---------------------------------------------------------------------------
# Mutation correctness
# ---------------------------------------------------------------------------

class TestRandomRename:
    def test_renames_function(self):
        assert "def sum_list" not in random_rename(SIMPLE_CODE)

    def test_renames_parameter(self):
        assert "nums" not in random_rename(SIMPLE_CODE)

    def test_deterministic_with_seed(self):
        assert random_rename(SIMPLE_CODE, seed=1) == random_rename(SIMPLE_CODE, seed=1)

    def test_different_seeds_differ(self):
        assert random_rename(SIMPLE_CODE, seed=1) != random_rename(SIMPLE_CODE, seed=2)


class TestNeutralRename:
    def test_uses_neutral_names(self):
        result = neutral_rename(SIMPLE_CODE)
        assert "def compute(" in result
        assert "value" in result

    def test_original_fn_gone(self):
        assert "def sum_list" not in neutral_rename(SIMPLE_CODE)


class TestDemographicRename:
    def test_male_western(self):
        r = demographic_rename(SIMPLE_CODE, "male_western")
        assert "def john_calc(" in r and "michael" in r

    def test_female_western(self):
        r = demographic_rename(SIMPLE_CODE, "female_western")
        assert "def emily_calc(" in r and "jessica" in r

    def test_male_south_asian(self):
        r = demographic_rename(SIMPLE_CODE, "male_south_asian")
        assert "def raj_calc(" in r and "vikram" in r

    def test_female_south_asian(self):
        r = demographic_rename(SIMPLE_CODE, "female_south_asian")
        assert "def priya_calc(" in r and "ananya" in r

    def test_invalid_group_raises(self):
        try:
            demographic_rename(SIMPLE_CODE, "unknown_group")
            assert False, "should have raised ValueError"
        except ValueError:
            pass


class TestApplyAllMutations:
    def test_original_unchanged(self):
        assert apply_all_mutations(SIMPLE_CODE)["original"] == SIMPLE_CODE

    def test_returns_all_keys(self):
        result = apply_all_mutations(SIMPLE_CODE)
        for name in ALL_MUTATION_NAMES:
            assert name in result, f"Missing: {name}"


class TestMutationLabel:
    def test_original(self):
        assert mutation_label("original") == "original"

    def test_random(self):
        assert mutation_label("random") == "random rename"

    def test_demo(self):
        label = mutation_label("demo_male_western")
        assert "male" in label.lower() and "western" in label.lower()


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_int(self):      assert normalize("15")    == "15"
    def test_bool(self):     assert normalize("True")  == "True"
    def test_list(self):     assert normalize("[1,2,3]") == "[1, 2, 3]"
    def test_fallback(self): assert normalize("some_text") == "some_text"


class TestIsCorrect:
    def test_match(self):      assert is_correct("15", "15")
    def test_mismatch(self):   assert not is_correct("14", "15")
    def test_list_match(self): assert is_correct("[1, 2, 3, 4]", "[1, 2, 3, 4]")


class TestConsistencyVerdict:
    def test_both_correct(self):
        v = consistency_verdict("15", "15", "15")
        assert not v["inconsistent"] and v["type"] == "consistent"

    def test_orig_correct_mut_wrong(self):
        v = consistency_verdict("15", "14", "15")
        assert v["inconsistent"] and v["type"] == "correctness"

    def test_both_wrong_same(self):
        v = consistency_verdict("14", "14", "15")
        assert not v["inconsistent"]

    def test_invalidity(self):
        v = consistency_verdict("15", "API_ERROR:timeout", "15")
        assert v["inconsistent"] and v["type"] == "invalidity"


class TestDemographicInconsistency:
    def test_demo_inconsistent_when_neutral_ok(self):
        assert is_demographically_inconsistent("15", "15", "14", "15") is True

    def test_not_demo_inconsistent_when_neutral_also_wrong(self):
        assert is_demographically_inconsistent("15", "14", "14", "15") is False

    def test_all_correct(self):
        assert is_demographically_inconsistent("15", "15", "15", "15") is False


class TestSelfConsistencyFilter:
    def test_retained_when_agree(self):
        r = self_consistency_filter("15", "15", "15")
        assert r["retained"] and r["answer"] == "15" and r["correct"]

    def test_abstain_when_disagree(self):
        r = self_consistency_filter("15", "14", "15")
        assert not r["retained"] and r["answer"] == "ABSTAIN" and r["correct"] is None

    def test_retained_wrong(self):
        r = self_consistency_filter("14", "14", "15")
        assert r["retained"] and not r["correct"]
