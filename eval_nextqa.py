import sys
import os
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from download_nextqa import download_annotations, download_videos
download_annotations()
download_videos()

from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.arguments import *
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.utils.message import Message

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "tinyllava", "data", "nextqa", "val_descriptive.csv")
VIDEO_ROOT = os.path.join(os.path.dirname(__file__), "tinyllava", "data", "nextqa", "videos", "NExTVideo")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results_3b_nextqa.csv")

def get_frames(v_path, n=8):
	vid = cv2.VideoCapture(v_path)
	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	if total <= 0: return []
	step = total // n
	frames = []
	for i in range(n):
	    vid.set(cv2.CAP_PROP_POS_FRAMES, i * step)
	    success, frame = vid.read()
	    if success:
	      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	      frames.append(Image.fromarray(frame))
	vid.release()
	return frames

print("Loading TinyLLaVA-3.1B...")
results = load_pretrained_model(MODEL_PATH, attn_implementation="eager")

model, tok, img_proc = None, None, None
for item in results:
   if hasattr(item, 'cuda'):
      model = item
   elif "Tokenizer" in str(type(item)):
      tok = item
   elif "Processor" in str(type(item)) or "Image" in str(type(item)):
      img_proc = item

print("Moving model to GPU (float16)...")
model = model.half().cuda()

text_processor = TextPreprocess(tok, 'phi')
stop_str = text_processor.template.separator.apply()[1]


questions = pd.read_csv(DATA_ROOT)
print(f"DEBUG: Available columns are: {questions.columns.tolist()}")
out_data = []
print(f"Running inference on {len(questions)} samples...")

for i, row in tqdm(questions.iterrows(), total=len(questions)):
	v_id = str(row['video'])
	v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")
	if not os.path.exists(v_file):
	    continue

	video_frames = get_frames(v_file)
	if not video_frames:
	     continue

	pixel_values = [img_proc.preprocess(f, return_tensors='pt')['pixel_values'][0] for f in video_frames]
	pixel_values = torch.stack(pixel_values).cuda()

	qs = row['question']
	choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
	options_str = "\n".join(f"{i}: {c}" for i, c in enumerate(choices))
	question = (DEFAULT_IMAGE_TOKEN + "\n" + qs + "\n" + options_str +
	            "\nAnswer with the option number only (0-4).")

	msg = Message()
	msg.add_message(question)
	result = text_processor(msg.messages, mode='eval')
	input_ids = result['input_ids'].unsqueeze(0).to(model.device)
	images_tensor = pixel_values.to(device=model.device)

	with torch.inference_mode():
	    output = model.generate(
		input_ids,
		images=images_tensor,
		do_sample=False,
		max_new_tokens=16,
		use_cache=True,
		pad_token_id=tok.eos_token_id
	   )

	res = tok.batch_decode(output, skip_special_tokens=True)[0].strip()
	if res.endswith(stop_str):
	    res = res[:-len(stop_str)].strip()

	# Extract first digit 0-4 as predicted index
	pred_idx = next((int(c) for c in res if c in '01234'), -1)

	out_data.append({
	    'videoID': v_id,
	    'question': qs,
	    'prediction_text': res,
	    'predicted_idx': pred_idx,
	    'answer_idx': int(row['answer']),
	    'correct': pred_idx == int(row['answer']),
	})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
pd.DataFrame(out_data).to_csv(OUT_PATH, index=False)
print(f"Done. Results saved in {OUT_PATH}")


