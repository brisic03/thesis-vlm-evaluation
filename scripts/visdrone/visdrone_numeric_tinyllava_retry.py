"""
Step 2B variant — TinyLLaVA VisDrone numeric counting WITH retry-until-answered.

In the original run TinyLLaVA refused (said "Many" / "cannot give an exact
number") on 26.8% of images. Greedy decoding is deterministic, so re-asking the
same prompt would just repeat the refusal. We therefore ESCALATE on refusal:

  attempt 1 : original prompt, greedy           (reproduces the baseline answer)
  attempt 2 : firmer prompt, greedy             (deterministic but different text)
  attempt 3+: firmer prompt, sampled (temp>0)   (varies until a number appears)

Stop as soon as a number is parsed, or after --max-attempts. Log how many
attempts each image needed. Unparseable after all attempts -> still NaN (logged).

Output: results/step2/step2b_tinyllava_numeric_retry.csv
"""
import sys, os, re, math, argparse, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
sys.path.insert(0, REPO)
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
GT_CSV   = os.path.join(REPO, "results/step2/visdrone_counting_gt.csv")
OUT_PATH = os.path.join(REPO, "results/step2/step2b_tinyllava_numeric_retry.csv")

WORD_TO_NUM = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
    "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20}

def parse_count(text):
    m = re.search(r"\d+", str(text).replace(",", ""))
    if m: return int(m.group(0))
    for w, n in WORD_TO_NUM.items():
        if re.search(r"\b"+w+r"\b", str(text).lower()): return n
    return None

def prompt_base(c):
    return f"How many {c}s are in this image? Answer with a number only."
def prompt_firm(c):
    return (f"How many {c}s are in this image? You must answer with a single whole "
            f"number that is your best estimate. Do not say 'many', 'several', "
            f"'a lot', or 'cannot'. Reply with just the number.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print("Loading TinyLLaVA-3.1B …")
    parts = load_pretrained_model(MODEL_PATH, attn_implementation="eager", device_map=None)
    model = tok = ip = None
    for p in parts:
        if hasattr(p, "parameters"): model = p
        elif "Tokenizer" in str(type(p)): tok = p
        elif "Processor" in str(type(p)) or "Image" in str(type(p)): ip = p
    model = model.half().cuda(); model.resize_token_embeddings(max(len(tok), model.config.vocab_size)+100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
    tp = TextPreprocess(tok, "phi"); imp = ImagePreprocess(ip, model.config)
    stop = tp.template.separator.apply()[1]

    def query(img_t, prompt, sample, seed=0):
        msg = Message(); msg.add_message(f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
        ids = tp(msg.messages, mode="eval")["input_ids"].unsqueeze(0).to(model.device)
        sc = KeywordsStoppingCriteria([stop], tok, ids)
        gen = dict(images=img_t, max_new_tokens=24, use_cache=True,
                   pad_token_id=tok.pad_token_id, stopping_criteria=[sc])
        if sample:
            torch.manual_seed(seed)
            gen.update(do_sample=True, temperature=args.temp, top_p=0.9)
        else:
            gen.update(do_sample=False)
        with torch.inference_mode():
            o = model.generate(ids, **gen)
        d = tok.batch_decode(o, skip_special_tokens=True)[0].strip()
        return d[:-len(stop)].strip() if d.endswith(stop) else d

    gt = pd.read_csv(GT_CSV)
    if args.limit: gt = gt.head(args.limit)
    print(f"counting questions: {len(gt)}  | max_attempts={args.max_attempts}")

    rows = []
    for i, (_, r) in enumerate(tqdm(gt.iterrows(), total=len(gt))):
        if not os.path.exists(r["image_path"]): continue
        c = r["target_class"]
        img_t = imp(Image.open(r["image_path"]).convert("RGB")).unsqueeze(0).to(model.device, dtype=torch.float16)
        raws, pred = [], None
        for att in range(1, args.max_attempts + 1):
            if att == 1:
                raw = query(img_t, prompt_base(c), sample=False)
            elif att == 2:
                raw = query(img_t, prompt_firm(c), sample=False)
            else:
                raw = query(img_t, prompt_firm(c), sample=True, seed=1000 + i * 13 + att)
            raws.append(raw)
            pred = parse_count(raw)
            if pred is not None:
                break
        rows.append({
            "image": r["image"], "target_class": c, "gt_count": int(r["gt_count"]),
            "pred_count": pred, "parse_ok": pred is not None,
            "n_attempts": len(raws), "raw_first": raws[0], "raw_final": raws[-1],
            "raws": str(raws),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[df.parse_ok]
    err = ok.pred_count - ok.gt_count
    pfr = 100.0 * (1 - len(ok) / len(df))
    print(f"\nSaved {len(df)} rows -> {args.out}")
    print(f"\n=== STEP 2B (retry) — TinyLLaVA VisDrone numeric ===")
    print(f"parseable: {len(ok)}/{len(df)}  (parse-failure rate {pfr:.1f}%)")
    print(f"MAE={err.abs().mean():.2f}  RMSE={math.sqrt((err**2).mean()):.2f}  "
          f"bias={err.mean():+.2f}")
    print(f"attempts: mean={df.n_attempts.mean():.2f}  "
          f"answered-on-try-1={(df.n_attempts==1).sum()}  needed-retry={(df.n_attempts>1).sum()}")
    print("attempts distribution:", df.n_attempts.value_counts().sort_index().to_dict())

if __name__ == "__main__":
    main()
