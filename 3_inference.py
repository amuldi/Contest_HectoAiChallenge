# 3_inference.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
import timm

TEST_DIR = "test_data"
MODEL_PATH = "models/best_model.pth"
SUBMIT_PATH = "submission/submission.csv"

IMG_SIZE = 380
NUM_FRAMES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.backbone(x)
        logit = self.fc(feat)
        return logit.view(B, T).mean(dim=1)


def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = []

    for video in tqdm(os.listdir(TEST_DIR)):
        cap = cv2.VideoCapture(os.path.join(TEST_DIR, video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total-1, NUM_FRAMES).astype(int)

        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(frame))

        cap.release()

        x = torch.stack(frames).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        results.append([video, prob])

    df = pd.DataFrame(results, columns=["video", "prediction"])
    df.to_csv(SUBMIT_PATH, index=False)
    print("Submission saved:", SUBMIT_PATH)


if __name__ == "__main__":
    main()