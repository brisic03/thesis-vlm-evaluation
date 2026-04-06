import os
import zipfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "tinyllava", "data", "nextqa")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
CSV_PATH = os.path.join(DATA_DIR, "val_descriptive.csv")
ZIP_PATH = os.path.join(DATA_DIR, "videos.zip")


def download_annotations():
    if os.path.exists(CSV_PATH):
        print(f"Annotations already exist at {CSV_PATH}, skipping.")
        return

    print("Downloading MC annotations from lmms-lab/NExTQA...")
    import pandas as pd
    ds = load_dataset("lmms-lab/NExTQA", "MC", split="test")
    df = ds.to_pandas()

    # Filter to descriptive question types (DC, DL, DO)
    desc = df[df["type"].isin(["DC", "DL", "DO"])]
    os.makedirs(DATA_DIR, exist_ok=True)
    desc.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(desc)} descriptive questions ({desc['video'].nunique()} unique videos) to {CSV_PATH}")


def download_videos():
    if os.path.isdir(VIDEO_DIR) and any(f.endswith(".mp4") for f in os.listdir(VIDEO_DIR)):
        print(f"Videos already exist in {VIDEO_DIR}, skipping.")
        return

    if not os.path.exists(ZIP_PATH):
        print("Downloading videos.zip (6.49 GB) from lmms-lab/NExTQA...")
        hf_hub_download(
            repo_id="lmms-lab/NExTQA",
            filename="videos.zip",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("Download complete.")
    else:
        print(f"videos.zip already exists at {ZIP_PATH}, skipping download.")

    print(f"Extracting videos to {VIDEO_DIR}...")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(VIDEO_DIR)
    print(f"Extraction complete. Videos at {os.path.join(VIDEO_DIR, 'NExTVideo')}")


if __name__ == "__main__":
    download_annotations()
    download_videos()
    print("Done. Run eval_nextqa.py to start evaluation.")
