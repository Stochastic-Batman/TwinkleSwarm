import os
import cv2
import numpy as np


def load_image(file_name: str) -> np.ndarray:
    path = os.path.join('data/images', file_name)
    if not os.path.exists(path): raise FileNotFoundError(f"Image {file_name} not found in data/images")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_video(file_name: str) -> cv2.VideoCapture:
    path = os.path.join('data/videos', file_name)
    if not os.path.exists(path): raise FileNotFoundError(f"Video {file_name} not found in data/videos")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError(f"Could not open video {file_name}")
    return cap