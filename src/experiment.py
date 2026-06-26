"""
CLI experiment runner for demographic lexical mutation testing.
Uses 50 real CruxEval programs loaded dynamically from HuggingFace or GitHub.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python src/experiment.py

Options
-------
    --programs   all | sample_178,sample_425,...   (default: all)
    --mutations  all | original,random,demo_neutral,...  (default: all)
    --output     path to JSON results file  (default: results/results.json)
    --delay      seconds between API calls  (default: 0.5)
    --model      OpenAI model name          (default: gpt-4o)
    --limit      max programs to run        (default: all)

Examples
--------
    # Quick smoke test: 5 programs
    python src/experiment.py --limit 5

    # Only demographic mutations on 10 programs
    python src/experiment.py --limit 10 --mutations original,demo_neutral,demo_male_western,demo_female_south_asian

    # Full run
    python src/experiment.py --programs all --mutations all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from programs import PROGRAMS
from mutations import apply_all_mutations, mutation_label, ALL_MUTATION_NAMES
from oracle import (
    consistency_verdict,
    is_demographically_inconsistent,
    self_consistency_filter,
    is_correct,
)

SYSTEM_PROMPT = (
    "You are a Python execution expert. "
    "When asked to predict a function's return value, reply with ONLY the "
    "return value exactly as Python would print it — no explanation, no "
    "code fence, just the value. For example: True, 15, 'hello', [1, 2, 3]."
)


def build_prompt(code: str, inp: str, desc: str) -> str:
    return (
        f"Function description: {desc}\n\n"
        f"{code}\n\n"
        f"What does the function return when called with input: {inp}\n\n"
        "Reply with ONLY the return value."
    )


def query_llm(client: OpenAI, code: str, inp: str, desc: str, model: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=150,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(code, inp, desc)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"API_ERROR:{exc}"


def run_experiment(
    programs=None,
    mutations=None,
    output_path="results/results.json",
    delay=0.5,
    model="gpt-4o",
):
    client = OpenAI()

    selected_programs = PROGRAMS if programs is None else [
        p for p in PROGRAMS if p["id"] in programs
    ]
    selected_mutations = mutations or ALL_MUTATION_NAMES

    all_cases = []
    total = len(selected_programs) * len(selected_mutations)
    done  = 0

    for p in selected_programs:
        mutated = apply_all_mutations(p["code"])
        answers = {}

        for mut_name in selected_mutations:
            if mut_name not in mutated:
                print(f"  skip unknown mutation: {mut_name}")
                continue

            code  = mutated[mut_name]
            done += 1
            label = mutation_label(mut_name)
            print(f"[{done:>4}/{total}] {p['id']} / {label:<30}", end="", flush=True)

            ans     = query_llm(client, code, p["inp"], p["desc"], model)
            correct = is_correct(ans, p["exp"])
            answers[mut_name] = ans

            print(f" {'✓' if correct else '✗'}  got: {ans!r:<25} expected: {p['exp']}")

            all_cases.append({
                "prog_id":        p["id"],
                "mutation":       mut_name,
                "mutation_label": label,
                "desc":           p["desc"],
                "code":           code,
                "inp":            p["inp"],
                "exp":            p["exp"],
                "answer":         ans,
                "correct":        correct,
            })

            time.sleep(delay)

        # ── consistency verdicts ──────────────────────────────────────────────
        orig_ans    = answers.get("original")
        neutral_ans = answers.get("demo_neutral")

        for case in all_cases:
            if case["prog_id"] != p["id"] or case["mutation"] == "original":
                continue
            if orig_ans is None:
                continue
            v = consistency_verdict(orig_ans, case["answer"], p["exp"])
            case["inconsistent"]       = v["inconsistent"]
            case["inconsistency_type"] = v["type"]

            if neutral_ans is not None:
                case["demographically_inconsistent"] = is_demographically_inconsistent(
                    orig_ans, neutral_ans, case["answer"], p["exp"]
                )

        # ── self-consistency filter ───────────────────────────────────────────
        if orig_ans is not None and neutral_ans is not None:
            fr = self_consistency_filter(orig_ans, neutral_ans, p["exp"])
            for case in all_cases:
                if case["prog_id"] == p["id"] and case["mutation"] == "original":
                    case["filter_retained"] = fr["retained"]
                    case["filter_answer"]   = fr["answer"]
                    case["filter_correct"]  = fr["correct"]

    # ── summary stats ─────────────────────────────────────────────────────────
    non_orig   = [c for c in all_cases if c.get("mutation") != "original"]
    demo_cases = [
        c for c in non_orig
        if c.get("mutation", "").startswith("demo_")
        and c.get("mutation") != "demo_neutral"
    ]

    total_incon = sum(1 for c in non_orig   if c.get("inconsistent"))
    demo_incon  = sum(1 for c in demo_cases if c.get("demographically_inconsistent"))
    overall_acc = sum(1 for c in all_cases  if c.get("correct"))

    summary = {
        "model":                              model,
        "total_programs":                     len(selected_programs),
        "total_cases":                        len(all_cases),
        "overall_accuracy_pct":               round(overall_acc / max(len(all_cases), 1) * 100, 2),
        "total_inconsistencies":              total_incon,
        "inconsistency_rate_pct":             round(total_incon / max(len(non_orig), 1) * 100, 2),
        "demographic_inconsistencies":        demo_incon,
        "demographic_inconsistency_rate_pct": round(demo_incon / max(len(demo_cases), 1) * 100, 2),
    }

    # per-mutation breakdown
    per_mutation = {}
    for mut_name in selected_mutations:
        if mut_name == "original":
            continue
        cases = [c for c in non_orig if c.get("mutation") == mut_name]
        incon = sum(1 for c in cases if c.get("inconsistent"))
        per_mutation[mut_name] = {
            "label":             mutation_label(mut_name),
            "total":             len(cases),
            "inconsistent":      incon,
            "inconsistency_pct": round(incon / max(len(cases), 1) * 100, 2),
        }

    output = {"summary": summary, "per_mutation": per_mutation, "cases": all_cases}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── print summary ─────────────────────────────────────────────────────────
    width = 60
    print("\n" + "=" * width)
    print(f"Model:                    {model}")
    print(f"Programs tested:          {len(selected_programs)}")
    print(f"Total cases:              {len(all_cases)}")
    print(f"Overall accuracy:         {summary['overall_accuracy_pct']}%")
    print(f"Inconsistency rate:       {summary['inconsistency_rate_pct']}%  ({total_incon}/{len(non_orig)})")
    print(f"Demographic incon. rate:  {summary['demographic_inconsistency_rate_pct']}%  ({demo_incon}/{len(demo_cases)})")
    print("\nPer-mutation breakdown:")
    for mut_name in selected_mutations:
        if mut_name == "original" or mut_name not in per_mutation:
            continue
        s   = per_mutation[mut_name]
        bar = "█" * int(s["inconsistency_pct"] / 5)
        print(f"  {s['label']:<28} {s['inconsistency_pct']:5.1f}%  {bar}")
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Demographic lexical mutation experiment on CruxEval programs"
    )
    parser.add_argument(
        "--programs", default="all",
        help="all, or comma-separated CruxEval IDs e.g. sample_178,sample_425"
    )
    parser.add_argument(
        "--mutations", default="all",
        help="all, or comma-separated mutation names"
    )
    parser.add_argument("--output",  default="results/results.json")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="seconds between API calls (default 0.5)")
    parser.add_argument("--model",   default="gpt-4o")
    parser.add_argument("--limit",   type=int, default=None,
                        help="max number of programs to test (for quick runs)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    programs = None if args.programs == "all" else args.programs.split(",")
    mutations = None if args.mutations == "all" else args.mutations.split(",")

    prog_list = PROGRAMS if programs is None else [
        p for p in PROGRAMS if p["id"] in programs
    ]
    if args.limit:
        prog_list = prog_list[:args.limit]
    programs = [p["id"] for p in prog_list]

    run_experiment(
        programs=programs,
        mutations=mutations,
        output_path=args.output,
        delay=args.delay,
        model=args.model,
    )


if __name__ == "__main__":
    main()
