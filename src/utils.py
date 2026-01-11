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


def get_target_points(image: np.ndarray, num_drones: int) -> np.ndarray:
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack([c.squeeze() for c in contours])
    if len(all_points) < num_drones:
        raise ValueError("Not enough points in image for num_drones")
    indices = np.linspace(0, len(all_points) - 1, num_drones, dtype=int)
    targets_2d = all_points[indices]
    targets_3d = np.hstack([targets_2d, np.zeros((num_drones, 1))])
    targets_3d = targets_3d.astype(float)
    targets_3d -= np.mean(targets_3d, axis=0)
    return targets_3d


def get_text_targets(text: str, num_drones: int) -> np.ndarray:
    height, width = 200, 800
    image = np.zeros((height, width), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (50, 150), font, 5, 255, 10, cv2.LINE_AA)
    return get_target_points(image, num_drones)