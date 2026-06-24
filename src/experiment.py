"""
CLI experiment runner.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python src/experiment.py

Options
-------
    --programs   all | P1,P2,...     (default: all)
    --mutations  all | original,random,demo_neutral,...  (default: all)
    --output     path to JSON results file  (default: results/results.json)
    --delay      seconds between API calls  (default: 0.3)
    --model      OpenAI model name          (default: gpt-4o)

Example
-------
    python src/experiment.py --programs P1,P2,P3 --mutations original,random,demo_male_western
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from programs import PROGRAMS
from mutations import apply_all_mutations, mutation_label, ALL_MUTATION_NAMES
from oracle import consistency_verdict, is_demographically_inconsistent, self_consistency_filter


SYSTEM_PROMPT = (
    "You are a Python execution expert. "
    "When asked to predict a function's return value, reply with ONLY the "
    "return value exactly as Python would print it. "
    "No explanation, no code fence, just the value."
)


def build_prompt(code: str, inp: str, desc: str) -> str:
    return (
        f"Function description: {desc}\n\n"
        f"{code}\n\n"
        f"What does the function return when called with input: {inp}\n\n"
        "Reply with ONLY the return value."
    )


def query_llm(client: OpenAI, code: str, inp: str, desc: str,
              model: str) -> str:
    prompt = build_prompt(code, inp, desc)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"API_ERROR:{exc}"


def run_experiment(
    programs=None,
    mutations=None,
    output_path="results/results.json",
    delay=0.3,
    model="gpt-4o",
):
    client = OpenAI()

    selected_programs = programs or [p["id"] for p in PROGRAMS]
    selected_mutations = mutations or ALL_MUTATION_NAMES

    prog_map = {p["id"]: p for p in PROGRAMS}
    all_cases = []

    total = len(selected_programs) * len(selected_mutations)
    done = 0

    for pid in selected_programs:
        p = prog_map[pid]
        mutated = apply_all_mutations(p["code"])
        answers = {}

        for mut_name in selected_mutations:
            if mut_name not in mutated:
                print(f"  skip unknown mutation: {mut_name}")
                continue

            code = mutated[mut_name]
            done += 1
            label = mutation_label(mut_name)
            print(f"[{done:>3}/{total}] {pid} / {label:<30}", end="", flush=True)

            ans = query_llm(client, code, p["inp"], p["desc"], model)

            from oracle import is_correct
            correct = is_correct(ans, p["exp"])
            answers[mut_name] = ans
            status = "✓" if correct else "✗"
            print(f" {status}  got: {ans!r:<20}  expected: {p['exp']}")

            all_cases.append({
                "prog_id": pid,
                "mutation": mut_name,
                "mutation_label": label,
                "desc": p["desc"],
                "code": code,
                "inp": p["inp"],
                "exp": p["exp"],
                "answer": ans,
                "correct": correct,
            })

            time.sleep(delay)

        # compute consistency verdicts against original
        orig_ans = answers.get("original")
        neutral_ans = answers.get("demo_neutral")

        for case in all_cases:
            if case["prog_id"] != pid or case["mutation"] == "original":
                continue
            if orig_ans is None:
                continue
            verdict = consistency_verdict(orig_ans, case["answer"], p["exp"])
            case["inconsistent"] = verdict["inconsistent"]
            case["inconsistency_type"] = verdict["type"]

            if neutral_ans is not None:
                case["demographically_inconsistent"] = is_demographically_inconsistent(
                    orig_ans, neutral_ans, case["answer"], p["exp"]
                )
            else:
                case["demographically_inconsistent"] = None

        # compute mitigation filter for this program
        if orig_ans is not None and neutral_ans is not None:
            filter_result = self_consistency_filter(orig_ans, neutral_ans, p["exp"])
            for case in all_cases:
                if case["prog_id"] == pid and case["mutation"] == "original":
                    case["filter_retained"] = filter_result["retained"]
                    case["filter_answer"] = filter_result["answer"]
                    case["filter_correct"] = filter_result["correct"]

    # summary stats
    non_orig = [c for c in all_cases if c.get("mutation") != "original"]
    total_incon = sum(1 for c in non_orig if c.get("inconsistent"))
    demo_cases = [c for c in non_orig if c.get("mutation", "").startswith("demo_")
                  and c.get("mutation") != "demo_neutral"]
    demo_incon = sum(1 for c in demo_cases if c.get("demographically_inconsistent"))

    summary = {
        "total_cases": len(all_cases),
        "total_inconsistencies": total_incon,
        "inconsistency_rate_pct": round(total_incon / max(len(non_orig), 1) * 100, 2),
        "demographic_inconsistencies": demo_incon,
        "demographic_inconsistency_rate_pct": round(demo_incon / max(len(demo_cases), 1) * 100, 2),
        "overall_accuracy_pct": round(
            sum(1 for c in all_cases if c.get("correct")) / max(len(all_cases), 1) * 100, 2
        ),
    }

    output = {"summary": summary, "cases": all_cases}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Total cases:              {summary['total_cases']}")
    print(f"Overall accuracy:         {summary['overall_accuracy_pct']}%")
    print(f"Inconsistency rate:       {summary['inconsistency_rate_pct']}%")
    print(f"Demographic incon. rate:  {summary['demographic_inconsistency_rate_pct']}%")
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Demographic lexical mutation experiment")
    parser.add_argument("--programs", default="all",
                        help="all, or comma-separated IDs e.g. P1,P2,P3")
    parser.add_argument("--mutations", default="all",
                        help="all, or comma-separated mutation names")
    parser.add_argument("--output", default="results/results.json")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="seconds between API calls")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    programs = None if args.programs == "all" else args.programs.split(",")
    mutations = None if args.mutations == "all" else args.mutations.split(",")

    run_experiment(
        programs=programs,
        mutations=mutations,
        output_path=args.output,
        delay=args.delay,
        model=args.model,
    )


if __name__ == "__main__":
    main()
