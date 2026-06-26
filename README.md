# Trustware Project
# Demographic Lexical Mutations as Adversarial Inputs in Code LLMs
**Task 3 prototype — LLM Testing Assessment**

This project implements and extends two research papers:
- **MUCOCO** (Chua et al., 2026) — automated consistency testing of Code LLMs via semantic-preserving mutations
- **HInter** (Souani et al., 2025) — exposing hidden intersectional bias in LLMs via metamorphic testing

## Research hypothesis
MUCOCO shows lexical mutations (random renaming) produce the highest inconsistency rate among all mutation types. HInter shows demographic attribute substitutions expose bias invisible to atomic testing. This prototype asks: **do demographically loaded variable/function names behave like adversarial lexical mutations in Code LLMs?**

## What's in this repo
```
Trustware-project/
├── README.md
├── requirements.txt
├── demo/
│   └── index.html          # Standalone browser demo (no server needed)
├── src/
│   ├── programs.py          # Loads 50 real CruxEval programs from HuggingFace or GitHub
│   ├── mutations.py         # All mutation operators (word-boundary safe)
│   ├── oracle.py            # Correctness, consistency, and demographic inconsistency oracles
│   └── experiment.py        # Full experiment runner (CLI)
├── results/
│   └── .gitkeep
└── test/
    └── test_mutations.py
```

## Quick start

### Browser demo (no Python required)
Open `demo/index.html` in any browser. Calls the OpenAI API directly from the browser. You need an OpenAI API key.

### CLI experiment runner
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here

# Quick smoke test (5 programs)
python src/experiment.py --limit 5

# Full run
python src/experiment.py --programs all --mutations all --output results/results.json
```

### Run tests
```bash
python -m pytest test/
```

### Inspect the program bank
```bash
python src/programs.py          # prints first 5 CruxEval programs
python src/programs.py --all    # prints all 50
```

## Program bank
50 real programs loaded dynamically from [CruxEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) (Gu et al., ICML 2024).

`programs.py` tries HuggingFace Hub first (`datasets` library); if unavailable, it falls back automatically to fetching the JSONL directly from GitHub using only Python stdlib. No separate download script is needed.

Programs are stratified by minimum parameter name length (25 single-letter, 10 two-letter, 15 longer) to cover the full difficulty range for lexical mutation — single-letter params carry the highest mutation risk as they have zero name signal and are most likely to collide with Python keywords.

## Mutation operators

| Operator | Category | Rationale |
|---|---|---|
| Original | Baseline | Unmodified program — ground truth answer |
| Random rename | Lexical | MUCOCO baseline — highest inconsistency rate (16.28%) |
| Neutral rename | Lexical (control) | Isolates name plausibility from demographic signal |
| Male / Western | Demographic-lexical | HInter-inspired: `john_calc(michael, william)` |
| Female / Western | Demographic-lexical | `emily_calc(jessica, sarah)` |
| Male / South Asian | Demographic-lexical | `raj_calc(vikram, arjun)` |
| Female / South Asian | Demographic-lexical | `priya_calc(ananya, deepa)` |

**Key methodological contribution:** The neutral rename condition is not present in MUCOCO or HInter. It enables direct attribution: any inconsistency that appears in a demographic condition but not in the neutral condition is caused by demographic signal, not by the general loss of meaningful names.

## Oracles

| Oracle | Purpose |
|---|---|
| `is_correct` | Normalised string match against ground truth |
| `consistency_verdict` | Classifies (original, mutated) pair as consistent / correctness-inconsistent / invalidity-inconsistent |
| `is_demographically_inconsistent` | Demographic variant disagrees with original AND neutral variant agrees — isolates demographic signal |
| `self_consistency_filter` | Mitigation: abstain when original and neutral disagree; retain when they agree |

## How it improves on the base papers

| | MUCOCO | HInter | This prototype |
|---|---|---|---|
| Tests demographic names in code? | ✗ | Text only | ✓ |
| Neutral control condition? | ✗ | ✗ | ✓ |
| Attributes inconsistency to demographics specifically? | ✗ | Partially | ✓ |
| Mitigation built in? | Appendix only | ✗ | ✓ Self-consistency filter |
| Real CruxEval programs? | ✓ | ✗ | ✓ 50 programs, loaded live |

## References
- Chua et al. (2026). MUCOCO: Automated Consistency Testing of Code LLMs. arXiv:2604.19086
- Souani et al. (2025). HInter: Exposing Hidden Intersectional Bias in LLMs. arXiv:2503.11962
- Honarvar et al. (2025). TURBULENCE. ICST 2025.
- Rajan et al. (2024). KonTest. Findings of EMNLP 2024.
- Gu et al. (2024). CruxEval. ICML 2024.
