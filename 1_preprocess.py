# 1_preprocess.py
import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = "train_data/ff++_datacrop"
SAVE_DIR = "train_data/frame_index"
NUM_FRAMES = 8

os.makedirs(SAVE_DIR, exist_ok=True)

def get_frame_idxs(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= num_frames:
        return list(range(total))
    return np.linspace(0, total - 1, num_frames).astype(int).tolist()


for label in ["real", "fake"]:
    src_dir = os.path.join(DATA_DIR, label)
    dst_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    for video in tqdm(os.listdir(src_dir), desc=label):
        path = os.path.join(src_dir, video)
        idxs = get_frame_idxs(path, NUM_FRAMES)
        np.save(os.path.join(dst_dir, video.replace(".mp4", ".npy")), idxs)

print("Frame index preprocessing done.")