"""
VisDrone numeric counting in GRID mode (tiling for higher effective resolution).

The whole-image baseline crushes a ~1360px aerial photo to a 336/384 square, so
small objects fall below one vision patch. Grid mode splits the image into an
NxN grid, counts the target class in each tile separately (each tile is shown at
the full encoder resolution -> Nx higher effective resolution per object), then
SUMS the per-tile counts.

Per-tile reply is parsed to a number; an unparseable tile counts as 0 for the sum
but is logged (tile_parse_fail_rate) so we never silently hide refusals.

Boundary caveat: an object split across a tile edge may be double-counted or
missed; we have no coordinates to dedupe, so this is accepted noise and noted.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 scripts/visdrone/visdrone_grid_count.py --model tinyllava --grid 2
  CUDA_VISIBLE_DEVICES=1 .venv-mobilevlm/bin/python3 scripts/visdrone/visdrone_grid_count.py --model mobilevlm --grid 3
"""
import sys, os, re, math, argparse, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
GT_CSV = f"{REPO}/results/step2/visdrone_counting_gt.csv"

# correct plurals (fixes the "peoples"/"buss" bug in the whole-image prompt)
PLURAL = {"pedestrian":"pedestrians","people":"people","bicycle":"bicycles",
          "car":"cars","van":"vans","truck":"trucks","tricycle":"tricycles",
          "awning-tricycle":"awning-tricycles","bus":"buses","motor":"motors"}

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

def tiles(img, n):
    W, H = img.size
    out = []
    for r in range(n):
        for c in range(n):
            box = (int(c*W/n), int(r*H/n), int((c+1)*W/n), int((r+1)*H/n))
            out.append(img.crop(box))
    return out

def build_loader(model_name):
    if model_name == "tinyllava":
        sys.path.insert(0, REPO)
        from tinyllava.model.load_model import load_pretrained_model
        from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
        from tinyllava.data import TextPreprocess, ImagePreprocess
        from tinyllava.utils.message import Message
        from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
        parts = load_pretrained_model("TinyLLaVA/TinyLLaVA-3.1B", attn_implementation="eager", device_map=None)
        model = tok = ip = None
        for p in parts:
            if hasattr(p, "parameters"): model = p
            elif "Tokenizer" in str(type(p)): tok = p
            elif "Processor" in str(type(p)) or "Image" in str(type(p)): ip = p
        model = model.half().cuda(); model.resize_token_embeddings(max(len(tok), model.config.vocab_size)+100)
        tp = TextPreprocess(tok, "phi"); imp = ImagePreprocess(ip, model.config)
        stop = tp.template.separator.apply()[1]
        def ask(img, prompt, sample=False, seed=0):
            msg = Message(); msg.add_message(f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
            ids = tp(msg.messages, mode="eval")["input_ids"].unsqueeze(0).to(model.device)
            it = imp(img).unsqueeze(0).to(model.device, dtype=torch.float16)
            sc = KeywordsStoppingCriteria([stop], tok, ids)
            gen = dict(images=it, max_new_tokens=16, pad_token_id=tok.pad_token_id,
                       use_cache=True, stopping_criteria=[sc])
            if sample:
                torch.manual_seed(seed); gen.update(do_sample=True, temperature=0.7, top_p=0.9)
            else:
                gen.update(do_sample=False)
            with torch.inference_mode():
                o = model.generate(ids, **gen)
            d = tok.batch_decode(o, skip_special_tokens=True)[0].strip()
            return d[:-len(stop)].strip() if d.endswith(stop) else d
        return ask
    else:
        sys.path.insert(0, os.path.join(REPO, "MobileVLM"))
        from mobilevlm.model.mobilevlm import load_pretrained_model
        from mobilevlm.conversation import conv_templates, SeparatorStyle
        from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        disable_torch_init()
        tok, model, ip, _ = load_pretrained_model("mtgv/MobileVLM_V2-3B"); model = model.half().cuda()
        def ask(img, prompt, sample=False, seed=0):
            it = process_images([img], ip, model.config).to(model.device, dtype=torch.float16)
            conv = conv_templates["v1"].copy(); conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
            conv.append_message(conv.roles[1], None); fp = conv.get_prompt()
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            ids = tokenizer_image_token(fp, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            sc = KeywordsStoppingCriteria([stop], tok, ids)
            gen = dict(images=it, max_new_tokens=16, use_cache=True, stopping_criteria=[sc])
            if sample:
                torch.manual_seed(seed); gen.update(do_sample=True, temperature=0.7, top_p=0.9)
            else:
                gen.update(do_sample=False)
            with torch.inference_mode():
                o = model.generate(ids, **gen)
            d = tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            return d[:-len(stop)].strip() if d.endswith(stop) else d
        return ask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tinyllava","mobilevlm"])
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="retry each tile (firmer prompt then sampling) until it gives a "
                         "number, instead of counting a refused tile as 0")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = "_force" if a.force else ""
    out = a.out or f"{REPO}/results/step3/visdrone_grid{a.grid}x{a.grid}_{a.model}{suffix}.csv"

    print(f"Loading {a.model} … grid={a.grid}x{a.grid}")
    ask = build_loader(a.model)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    gt = pd.read_csv(GT_CSV)
    if a.limit: gt = gt.head(a.limit)
    rows = []
    tile_fail = 0; tile_tot = 0; tile_forced = 0
    for ti, (_, r) in enumerate(tqdm(gt.iterrows(), total=len(gt))):
        if not os.path.exists(r["image_path"]): continue
        img = Image.open(r["image_path"]).convert("RGB")
        plural = PLURAL[r["target_class"]]
        p_base = f"How many {plural} are in this image? Answer with a number only."
        p_firm = (f"How many {plural} are in this image? You must answer with a single "
                  f"whole number that is your best estimate. Do not say 'many', 'several', "
                  f"or 'cannot'. Reply with just the number.")
        per_tile = []
        for tj, t in enumerate(tiles(img, a.grid)):
            tile_tot += 1
            v = parse_count(ask(t, p_base))
            if v is None and a.force:
                # escalate: firmer prompt greedy, then sampled retries
                for att in range(2, a.max_attempts + 1):
                    if att == 2:
                        v = parse_count(ask(t, p_firm))
                    else:
                        v = parse_count(ask(t, p_firm, sample=True, seed=1000 + ti * 37 + tj * 7 + att))
                    if v is not None:
                        tile_forced += 1
                        break
            if v is None:
                tile_fail += 1
            per_tile.append(v if v is not None else 0)
        rows.append({"image": r["image"], "target_class": r["target_class"],
                     "gt_count": int(r["gt_count"]), "pred_count": sum(per_tile),
                     "tile_counts": str(per_tile)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    err = df.pred_count - df.gt_count
    print(f"\nSaved {len(df)} rows -> {out}")
    print(f"=== VisDrone grid {a.grid}x{a.grid} — {a.model}{'  [FORCE]' if a.force else ''} ===")
    print(f"MAE={err.abs().mean():.2f}  RMSE={math.sqrt((err**2).mean()):.2f}  bias={err.mean():+.2f}")
    print(f"tile parse-fail (still 0 after retries): {tile_fail}/{tile_tot} ({100*tile_fail/tile_tot:.1f}%)")
    if a.force:
        print(f"tiles forced (refused then answered on retry): {tile_forced}/{tile_tot} ({100*tile_forced/tile_tot:.1f}%)")
    print(f"pred mean={df.pred_count.mean():.1f}  gt mean={df.gt_count.mean():.1f}")

if __name__ == "__main__":
    main()
