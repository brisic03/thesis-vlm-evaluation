"""
Step 2B — TinyLLaVA-3.1B open-ended numeric counting on VisDrone.

Prompt (per image, single frame):
  "How many <class>s are in this image? Answer with a number only."
Parse an integer from the reply; UNPARSEABLE -> NaN (logged as parse-failure,
never silently treated as 0). Metrics computed over parseable rows: MAE, RMSE.

Input : results/step2/visdrone_counting_gt.csv  (image_path, target_class, gt_count)
Output: results/step2/step2b_tinyllava_numeric.csv
"""

import sys
import os
import re
import math
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
sys.path.insert(0, REPO_ROOT)

from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
GT_CSV     = os.path.join(REPO_ROOT, "results/step2/visdrone_counting_gt.csv")
OUT_PATH   = os.path.join(REPO_ROOT, "results/step2/step2b_tinyllava_numeric.csv")

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_count(text):
    """First integer in the reply, else first number-word, else None."""
    m = re.search(r"\d+", text.replace(",", ""))
    if m:
        return int(m.group(0))
    for w, n in WORD_TO_NUM.items():
        if re.search(r"\b" + w + r"\b", text.lower()):
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print("Loading TinyLLaVA-3.1B …")
    parts = load_pretrained_model(MODEL_PATH, attn_implementation="eager", device_map=None)
    model = tok = img_proc = None
    for p in parts:
        if hasattr(p, "parameters"):
            model = p
        elif "Tokenizer" in str(type(p)):
            tok = p
        elif "Processor" in str(type(p)) or "Image" in str(type(p)):
            img_proc = p
    model = model.half().cuda()
    model.resize_token_embeddings(max(len(tok), model.config.vocab_size) + 100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
    text_proc  = TextPreprocess(tok, "phi")
    image_proc = ImagePreprocess(img_proc, model.config)
    stop_str   = text_proc.template.separator.apply()[1]

    def query(image, target_class):
        prompt = (f"{DEFAULT_IMAGE_TOKEN}\n"
                  f"How many {target_class}s are in this image? "
                  f"Answer with a number only.")
        msg = Message()
        msg.add_message(prompt)
        input_ids = text_proc(msg.messages, mode="eval")["input_ids"].unsqueeze(0).to(model.device)
        img_t = image_proc(image).unsqueeze(0).to(model.device, dtype=torch.float16)
        stopping = KeywordsStoppingCriteria([stop_str], tok, input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=img_t, do_sample=False, max_new_tokens=16,
                pad_token_id=tok.pad_token_id, use_cache=True,
                stopping_criteria=[stopping])
        decoded = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        if decoded.endswith(stop_str):
            decoded = decoded[: -len(stop_str)].strip()
        return decoded

    gt = pd.read_csv(GT_CSV)
    if args.limit:
        gt = gt.head(args.limit)
    print(f"counting questions: {len(gt)}")

    rows = []
    for _, r in tqdm(gt.iterrows(), total=len(gt)):
        if not os.path.exists(r["image_path"]):
            continue
        image = Image.open(r["image_path"]).convert("RGB")
        raw = query(image, r["target_class"])
        pred = parse_count(raw)
        rows.append({
            "image": r["image"],
            "target_class": r["target_class"],
            "gt_count": int(r["gt_count"]),
            "raw_output": raw,
            "pred_count": pred,
            "parse_ok": pred is not None,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[df["parse_ok"]].copy()
    err = (ok["pred_count"] - ok["gt_count"]).abs()
    mae = err.mean()
    rmse = math.sqrt(((ok["pred_count"] - ok["gt_count"]) ** 2).mean())
    pfr = 100.0 * (1 - ok.shape[0] / df.shape[0])
    print(f"\nSaved {len(df)} rows -> {args.out}")
    print(f"\n=== STEP 2B — TinyLLaVA VisDrone numeric ===")
    print(f"parseable: {len(ok)}/{len(df)}  (parse-failure rate {pfr:.1f}%)")
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")


if __name__ == "__main__":
    main()
