"""
Step 4 — symbolic repair on NExT-QA counting (CSV-only, no GPU).

Idea: instead of voting on the per-frame letters, read the NUMBER the model
stated in each frame (saved in Step 1), combine those numbers with a simple
fixed rule (mode / median / max), and map the result to the nearest answer
option. Compare to (a) the per-frame letter vote and (b) the composite baseline.

Per-frame number = the EARLIEST number mentioned in that frame's text
(digit or number-word). Frames stating no number are skipped for that question.

Inputs:
  results/step1/step1_<model>_reasoning.csv      (per-frame stated text)
  results_<model>_3b_nextqa.csv                  (per-frame letter vote)
  results/step3/step3a_<model>_framesweep.csv    (composite, use n_frames==8)
Outputs:
  results/step4/step4_<model>_repair.csv         (per-question detail)
  results/step4/step4_summary.csv                (accuracy table)
"""

import os
import re
import ast
import statistics
from collections import Counter
import pandas as pd

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"

MODELS = {
    "TinyLLaVA": {
        "step1":    os.path.join(REPO, "results/step1/step1_tinyllava_reasoning.csv"),
        "vote":     os.path.join(REPO, "results_3b_nextqa.csv"),
        "sweep":    os.path.join(REPO, "results/step3/step3a_tinyllava_framesweep.csv"),
    },
    "MobileVLM": {
        "step1":    os.path.join(REPO, "results/step1/step1_mobilevlm_reasoning.csv"),
        "vote":     os.path.join(REPO, "results_mobilevlm_3b_nextqa.csv"),
        "sweep":    os.path.join(REPO, "results/step3/step3a_mobilevlm_framesweep.csv"),
    },
}

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
# number token = a digit run OR a spelled-out word (longest words first so e.g.
# "seventeen" is matched before "seven")
_NUMWORDS = sorted(WORD_TO_NUM, key=len, reverse=True)
NUMTOK = r"(?:\d+|" + "|".join(_NUMWORDS) + r")"


def _tok2int(s):
    s = s.strip().lower()
    return WORD_TO_NUM[s] if s in WORD_TO_NUM else int(s)


def question_noun(question):
    """The object being counted, e.g. 'how many goats can ...' -> 'goats'."""
    m = re.search(r"how many\s+([a-z][a-z\-]*)", str(question).lower())
    return m.group(1) if m else None


def _all_numbers(low):
    """[(pos, value)] for every number token in the text, in order."""
    out = []
    for m in re.finditer(r"\d+", low):
        out.append((m.start(), int(m.group())))
    for w, n in WORD_TO_NUM.items():
        for m in re.finditer(r"\b" + w + r"\b", low):
            out.append((m.start(), n))
    return sorted(out)


def extract_count(text, noun=None):
    """
    Pull the stated count using several regex strategies, most reliable first:
      1. the number directly attached to the asked object ("5 people"/"three cars")
      2. an explicit count phrase ("there are N", "I see N", "count is N")
      3. the whole reply is a bare number ("5", "zero")
      4. the LAST number (the Step 1 prompt asks to state the count at the end)
      5. the FIRST number anywhere (last resort)
    Returns an int or None.
    """
    text = str(text)
    low = text.lower()

    # 1. "<num> <noun>" — the count of the specific object (try plural + singular)
    if noun:
        variants = {noun, re.sub(r"s$", "", noun)}
        for nv in variants:
            if not nv:
                continue
            m = re.search(NUMTOK + r"\s+" + re.escape(nv), low)
            if m:
                return _tok2int(m.group(0).split()[0])

    # 2. explicit count phrases
    for pat in (
        r"there (?:are|is|were|was)\s+(?:about |around |approximately |roughly |only )?(" + NUMTOK + r")",
        r"(?:i (?:can )?see|i count|counted|count is|total of|a total of|count:?)\s+(" + NUMTOK + r")",
    ):
        m = re.search(pat, low)
        if m:
            return _tok2int(m.group(1))

    # 3. the reply is just a number
    st = text.strip().lower().rstrip(".")
    if re.fullmatch(r"\d+", st):
        return int(st)
    if st in WORD_TO_NUM:
        return WORD_TO_NUM[st]

    # 4 & 5. positional fallbacks: last number, then first
    nums = _all_numbers(low)
    if nums:
        return nums[-1][1]
    return None


def option_to_num(text):
    t = str(text).strip().lower()
    if t in WORD_TO_NUM:
        return WORD_TO_NUM[t]
    try:
        return int(t)
    except ValueError:
        return None


def map_to_option(value, options):
    """Nearest numeric option to `value`; tie -> smaller count. None if no numeric opt."""
    cands = [(L, option_to_num(t)) for L, t in options.items()]
    cands = [(L, n) for L, n in cands if n is not None]
    if not cands:
        return None
    cands.sort(key=lambda x: (abs(x[1] - value), x[1]))
    return cands[0][0]


def mode_smallest(nums):
    c = Counter(nums)
    top = max(c.values())
    return min(v for v, k in c.items() if k == top)


def key(df):
    return df["videoID"].astype(str) + "||" + df["question"].astype(str)


def main():
    summary = []
    for model, paths in MODELS.items():
        s1 = pd.read_csv(paths["step1"])
        s1["_k"] = key(s1)

        # baseline letter vote on the same how-many subset
        vote = pd.read_csv(paths["vote"])
        vote["_k"] = key(vote)
        vote_hw = vote[vote["question"].str.lower().str.startswith("how many")]
        vote_map = dict(zip(vote_hw["_k"], vote_hw["correct"]))

        # composite @ 8 frames
        comp_map = {}
        if os.path.exists(paths["sweep"]):
            sw = pd.read_csv(paths["sweep"])
            sw8 = sw[sw["n_frames"] == 8].copy()
            sw8["_k"] = key(sw8)
            comp_map = dict(zip(sw8["_k"], sw8["correct"]))

        rows = []
        for _, r in s1.iterrows():
            options = ast.literal_eval(r["options"])
            gt = str(r["answer_letter"]).strip().upper()
            texts = ast.literal_eval(r["frame_reasoning"])
            noun = question_noun(r["question"])
            nums = [extract_count(t, noun) for t in texts]
            nums = [n for n in nums if n is not None]

            rec = {"videoID": str(r["videoID"]), "question": r["question"],
                   "answer_letter": gt, "frame_numbers": str(nums),
                   "_k": r["_k"]}
            if nums:
                preds = {
                    "mode":   map_to_option(mode_smallest(nums), options),
                    "median": map_to_option(statistics.median(nums), options),
                    "max":    map_to_option(max(nums), options),
                }
            else:
                preds = {"mode": None, "median": None, "max": None}
            for rule, letter in preds.items():
                rec[f"{rule}_pred"] = letter
                rec[f"{rule}_correct"] = int(letter == gt)
            # oracle: does ANY frame's number map to the correct option?
            frame_letters = {map_to_option(n, options) for n in nums}
            rec["oracle_anyframe_correct"] = int(gt in frame_letters)
            rows.append(rec)

        det = pd.DataFrame(rows)
        os.makedirs(os.path.join(REPO, "results/step4"), exist_ok=True)
        det.drop(columns=["_k"]).to_csv(
            os.path.join(REPO, f"results/step4/step4_{model.lower()}_repair.csv"), index=False)

        n = len(det)
        # align vote/composite to the same questions
        vote_acc = 100.0 * det["_k"].map(vote_map).fillna(0).mean()
        comp_acc = 100.0 * det["_k"].map(comp_map).fillna(0).mean() if comp_map else None
        summary.append({"model": model, "method": "per-frame vote (baseline)", "n": n,
                        "accuracy": round(vote_acc, 1)})
        if comp_acc is not None:
            summary.append({"model": model, "method": "composite @ 8 frames", "n": n,
                            "accuracy": round(comp_acc, 1)})
        for rule in ["mode", "median", "max"]:
            acc = 100.0 * det[f"{rule}_correct"].mean()
            summary.append({"model": model, "method": f"symbolic repair: {rule}", "n": n,
                            "accuracy": round(acc, 1)})
        orc = 100.0 * det["oracle_anyframe_correct"].mean()
        summary.append({"model": model, "method": "oracle: best single frame (ceiling)",
                        "n": n, "accuracy": round(orc, 1)})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(REPO, "results/step4/step4_summary.csv"), index=False)

    print("=== STEP 4 — symbolic repair vs vote vs composite (NExT-QA counting, n=177) ===\n")
    for model in sdf["model"].unique():
        sub = sdf[sdf["model"] == model]
        print(f"{model}:")
        for _, r in sub.iterrows():
            print(f"   {r['method']:<38} {r['accuracy']:>5}%")
        print()


if __name__ == "__main__":
    main()
