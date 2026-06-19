"""
Step 2B — MobileVLM V2-3B open-ended numeric counting on VisDrone.

Mirror of the TinyLLaVA numeric script. Prompt per image:
  "How many <class>s are in this image? Answer with a number only."
Parse an integer; UNPARSEABLE -> NaN (logged, never treated as 0).

Input : results/step2/visdrone_counting_gt.csv
Output: results/step2/step2b_mobilevlm_numeric.csv
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

REPO_ROOT      = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
MOBILEVLM_REPO = os.path.join(REPO_ROOT, "MobileVLM")
sys.path.insert(0, MOBILEVLM_REPO)

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import (
    disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria,
)
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "mtgv/MobileVLM_V2-3B"
GT_CSV     = os.path.join(REPO_ROOT, "results/step2/visdrone_counting_gt.csv")
OUT_PATH   = os.path.join(REPO_ROOT, "results/step2/step2b_mobilevlm_numeric.csv")
CONV_MODE  = "v1"

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_count(text):
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

    print("Loading MobileVLM V2-3B …")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH)
    model = model.half().cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    def query(image, target_class):
        prompt = (f"{DEFAULT_IMAGE_TOKEN}\n"
                  f"How many {target_class}s are in this image? "
                  f"Answer with a number only.")
        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16)
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=images_tensor, do_sample=False, max_new_tokens=16,
                use_cache=True, stopping_criteria=[stopping])
        decoded = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
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
    print(f"\n=== STEP 2B — MobileVLM VisDrone numeric ===")
    print(f"parseable: {len(ok)}/{len(df)}  (parse-failure rate {pfr:.1f}%)")
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")


if __name__ == "__main__":
    main()
