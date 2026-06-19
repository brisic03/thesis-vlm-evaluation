"""Step 0 of the eval plan: characterise VisDrone question/answer distributions.

Pure CSV analysis, no model runs, no third-party deps (stdlib csv only).
Computes, per the plan:
  (a) counting answer distribution over buckets 0,1,2,3,"4 or more"
  (b) location answer distribution (fraction "center")
  (c) majority-class baseline accuracy for counting and location

Gate: if any counting bucket > 60%, the multiple-choice counting result is not
defensible -> move VisDrone counting to numeric redesign (Step 2).
"""
import csv
import os
from collections import Counter

QUESTIONS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results", "visdrone", "visdrone_val_questions.csv",
)

COUNTING_BUCKETS = ["0", "1", "2", "3", "4 or more"]


def load_by_qtype(path):
    rows_by_type = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_type.setdefault(row["qtype"], []).append(row)
    return rows_by_type


def distribution(rows, field="answer_text", order=None):
    counts = Counter(r[field] for r in rows)
    total = sum(counts.values())
    keys = order if order is not None else sorted(counts, key=lambda k: -counts[k])
    # include any unexpected keys not in a provided order
    if order is not None:
        keys = list(order) + [k for k in counts if k not in order]
    return total, [(k, counts.get(k, 0), counts.get(k, 0) / total if total else 0.0)
                   for k in keys]


def majority_baseline(rows, field="answer_text"):
    counts = Counter(r[field] for r in rows)
    total = sum(counts.values())
    top_key, top_n = counts.most_common(1)[0]
    return top_key, top_n / total if total else 0.0


def print_table(title, total, dist):
    print(f"\n{title}  (n={total})")
    print(f"  {'answer':<12} {'count':>7} {'fraction':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*10}")
    for key, n, frac in dist:
        print(f"  {key:<12} {n:>7} {frac:>9.1%}")


def main():
    rows_by_type = load_by_qtype(os.path.abspath(QUESTIONS_CSV))
    print("qtype counts:", {k: len(v) for k, v in rows_by_type.items()})

    # (a) counting distribution
    counting = rows_by_type.get("counting", [])
    c_total, c_dist = distribution(counting, order=COUNTING_BUCKETS)
    print_table("(a) COUNTING answer distribution", c_total, c_dist)

    # (b) location distribution
    location = rows_by_type.get("location", [])
    l_total, l_dist = distribution(location)
    print_table("(b) LOCATION answer distribution", l_total, l_dist)

    # (c) majority-class baselines
    print("\n(c) Majority-class baseline accuracy")
    for name, rows in [("counting", counting), ("location", location)]:
        if not rows:
            continue
        key, acc = majority_baseline(rows)
        print(f"  {name:<10} always-pick '{key}'  ->  {acc:.1%}")

    # Gate check
    if c_total:
        worst = max(frac for _, _, frac in c_dist)
        worst_key = max(c_dist, key=lambda t: t[2])[0]
        print("\nGATE (counting): >60% in one bucket?")
        if worst > 0.60:
            print(f"  YES — '{worst_key}' = {worst:.1%}  ->  NOT defensible; "
                  f"move VisDrone counting to numeric redesign (Step 2).")
        else:
            print(f"  NO — largest bucket '{worst_key}' = {worst:.1%} (<=60%).")


if __name__ == "__main__":
    main()
