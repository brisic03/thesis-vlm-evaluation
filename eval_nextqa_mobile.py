import torch
import cv2
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import tokenizer_image_token, get_model_name_from_path 
from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3B"
DATA_ROOT = "/home/brisic03/dataset/nextqa/val_descriptive.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/database/videos/val"
OUT_PATH = "results_3b_nextqa.csv"

def get_frames(v_path, n=8):
	vid = cv2.VideCapture(v_path)
	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	step = total // n
	frames = []
	for i in range(n):
	    vid.set(cv2.CAP_PROP_POS_FRAMES, i * step)
