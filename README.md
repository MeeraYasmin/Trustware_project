# Trustware_project

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
│   ├── mutations.py         # All mutation operators (Python)
│   ├── oracle.py            # Correctness + consistency oracles
│   ├── programs.py          # Program bank (8 CruxEval-style programs)
│   └── experiment.py        # Full experiment runner (CLI)
├── results/
│   └── .gitkeep
└── test/
    └── test_mutations.py
```

## Quick start

### Browser demo (no Python required)
Open `demo/index.html` in any browser. Calls the Anthropic API directly from the browser. You need an Anthropic API key.

### CLI experiment runner
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python src/experiment.py --programs all --mutations all
```

### Run tests
```bash
python -m pytest tests/
```

## Mutation operators

| Operator | Category | Rationale |
|---|---|---|
| Random rename | Lexical | MUCOCO baseline — highest inconsistency rate |
| Neutral rename | Lexical (control) | Isolates name plausibility from demographic signal |
| Male/western | Demographic-lexical | HInter-inspired: `john_calc`, `michael` |
| Female/western | Demographic-lexical | `emily_calc`, `jessica` |
| Male/South Asian | Demographic-lexical | `raj_calc`, `vikram` |
| Female/South Asian | Demographic-lexical | `priya_calc`, `ananya` |

## References

- Chua et al. (2026). MUCOCO: Automated Consistency Testing of Code LLMs. arXiv:2604.19086
- Souani et al. (2025). HInter: Exposing Hidden Intersectional Bias in LLMs. arXiv:2503.11962
- Honarvar et al. (2025). TURBULENCE. ICST 2025.
- Rajan et al. (2024). KonTest. Findings of EMNLP 2024.
- Sap et al. (2020). Social Bias Frames. ACL 2020.
- Gu et al. (2024). CruxEval. ICML 2024.
