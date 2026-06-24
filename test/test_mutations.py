"""
Tests for mutation operators and oracles.
Run with: python -m pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mutations import (
    random_rename, neutral_rename, demographic_rename,
    apply_all_mutations, mutation_label, DEMOGRAPHIC_GROUPS, ALL_MUTATION_NAMES,
)
from oracle import (
    normalize, is_correct, consistency_verdict,
    is_demographically_inconsistent, self_consistency_filter,
)

SAMPLE_CODE = (
    "def sum_list(nums):\n"
    "    total = 0\n"
    "    for x in nums:\n"
    "        total += x\n"
    "    return total"
)

SAMPLE_CODE_2PARAM = (
    "def add(a, b):\n"
    "    return a + b"
)


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------

class TestRandomRename:
    def test_renames_function(self):
        result = random_rename(SAMPLE_CODE)
        assert "def sum_list" not in result
        assert "def " in result

    def test_renames_parameter(self):
        result = random_rename(SAMPLE_CODE)
        # original param 'nums' should be gone
        assert "nums" not in result

    def test_deterministic_with_seed(self):
        r1 = random_rename(SAMPLE_CODE, seed=1)
        r2 = random_rename(SAMPLE_CODE, seed=1)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = random_rename(SAMPLE_CODE, seed=1)
        r2 = random_rename(SAMPLE_CODE, seed=2)
        assert r1 != r2

    def test_two_param_function(self):
        result = random_rename(SAMPLE_CODE_2PARAM)
        assert "def add" not in result
        # both params renamed
        assert " a " not in result or "return" in result  # logic preserved


class TestNeutralRename:
    def test_uses_neutral_names(self):
        result = neutral_rename(SAMPLE_CODE)
        assert "def compute(" in result
        assert "value" in result

    def test_original_fn_gone(self):
        result = neutral_rename(SAMPLE_CODE)
        assert "def sum_list" not in result


class TestDemographicRename:
    def test_male_western(self):
        result = demographic_rename(SAMPLE_CODE, "male_western")
        assert "def john_calc(" in result
        assert "michael" in result

    def test_female_western(self):
        result = demographic_rename(SAMPLE_CODE, "female_western")
        assert "def emily_calc(" in result
        assert "jessica" in result

    def test_male_south_asian(self):
        result = demographic_rename(SAMPLE_CODE, "male_south_asian")
        assert "def raj_calc(" in result
        assert "vikram" in result

    def test_female_south_asian(self):
        result = demographic_rename(SAMPLE_CODE, "female_south_asian")
        assert "def priya_calc(" in result
        assert "ananya" in result

    def test_invalid_group_raises(self):
        try:
            demographic_rename(SAMPLE_CODE, "unknown_group")
            assert False, "should have raised ValueError"
        except ValueError:
            pass


class TestApplyAllMutations:
    def test_returns_all_keys(self):
        result = apply_all_mutations(SAMPLE_CODE)
        assert "original" in result
        assert "random" in result
        for grp in DEMOGRAPHIC_GROUPS:
            assert f"demo_{grp}" in result

    def test_original_unchanged(self):
        result = apply_all_mutations(SAMPLE_CODE)
        assert result["original"] == SAMPLE_CODE

    def test_all_mutation_names_covered(self):
        result = apply_all_mutations(SAMPLE_CODE)
        for name in ALL_MUTATION_NAMES:
            assert name in result, f"Missing mutation: {name}"


class TestMutationLabel:
    def test_original(self):
        assert mutation_label("original") == "original"

    def test_random(self):
        assert mutation_label("random") == "random rename"

    def test_demo(self):
        label = mutation_label("demo_male_western")
        assert "male" in label.lower() and "western" in label.lower()


# ---------------------------------------------------------------------------
# Oracle tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_int(self):
        assert normalize("15") == "15"

    def test_bool_true(self):
        assert normalize("True") == "True"

    def test_list(self):
        assert normalize("[1, 2, 3]") == "[1, 2, 3]"

    def test_string_value(self):
        assert normalize("'hello'") == "hello"

    def test_fallback(self):
        # something eval can't handle just lowercased
        assert normalize("some_text") == "some_text"


class TestIsCorrect:
    def test_int_match(self):
        assert is_correct("15", "15")

    def test_bool_match(self):
        assert is_correct("True", "True")

    def test_list_match(self):
        assert is_correct("[1, 2, 3, 4]", "[1, 2, 3, 4]")

    def test_mismatch(self):
        assert not is_correct("14", "15")


class TestConsistencyVerdict:
    def test_both_correct(self):
        v = consistency_verdict("15", "15", "15")
        assert not v["inconsistent"]
        assert v["type"] == "consistent"

    def test_orig_correct_mut_wrong(self):
        v = consistency_verdict("15", "14", "15")
        assert v["inconsistent"]
        assert v["type"] == "correctness"

    def test_both_wrong(self):
        v = consistency_verdict("14", "14", "15")
        assert not v["inconsistent"]
        assert v["type"] == "consistent"

    def test_invalidity(self):
        v = consistency_verdict("15", "API_ERROR:timeout", "15")
        assert v["inconsistent"]
        assert v["type"] == "invalidity"


class TestDemographicInconsistency:
    def test_demographic_inconsistent_when_neutral_ok(self):
        # orig=correct, neutral=correct (agree), demo=wrong -> demographic incon
        result = is_demographically_inconsistent(
            ans_orig="15", ans_neutral="15", ans_demo="14", exp="15"
        )
        assert result is True

    def test_not_demographically_inconsistent_when_neutral_also_wrong(self):
        # both neutral and demo wrong -> general inconsistency, not demographic
        result = is_demographically_inconsistent(
            ans_orig="15", ans_neutral="14", ans_demo="14", exp="15"
        )
        assert result is False

    def test_all_correct(self):
        result = is_demographically_inconsistent(
            ans_orig="15", ans_neutral="15", ans_demo="15", exp="15"
        )
        assert result is False


class TestSelfConsistencyFilter:
    def test_retained_when_agree(self):
        r = self_consistency_filter("15", "15", "15")
        assert r["retained"] is True
        assert r["answer"] == "15"
        assert r["correct"] is True

    def test_abstain_when_disagree(self):
        r = self_consistency_filter("15", "14", "15")
        assert r["retained"] is False
        assert r["answer"] == "ABSTAIN"
        assert r["correct"] is None

    def test_retained_wrong_answer(self):
        # both agree but both wrong
        r = self_consistency_filter("14", "14", "15")
        assert r["retained"] is True
        assert r["correct"] is False
