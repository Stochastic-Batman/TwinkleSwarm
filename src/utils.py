import os
import cv2
import numpy as np


np.random.seed(95)  # ⚡


def load_image(file_name: str) -> np.ndarray:
    path = os.path.join('data/images', file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image {file_name} not found in data/images")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_video(file_name: str) -> cv2.VideoCapture:
    path = os.path.join('data/videos', file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video {file_name} not found in data/videos")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {file_name}")
    return cap


def get_target_points(image: np.ndarray, num_drones: int) -> np.ndarray:
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack([c.squeeze() for c in contours if len(c) > 0])
    if len(all_points) < num_drones:
        raise ValueError(f"Not enough points in image for {num_drones} drones, found only {len(all_points)}")
    indices = np.linspace(0, len(all_points) - 1, num_drones, dtype=int)
    targets_2d = all_points[indices]
    targets_3d = np.hstack([targets_2d, np.zeros((num_drones, 1))]).astype(float)
    targets_3d -= np.mean(targets_3d, axis=0)
    targets_3d[:, 0] = (targets_3d[:, 0] / image.shape[1]) * 50
    targets_3d[:, 1] = (targets_3d[:, 1] / image.shape[0]) * 50
    return targets_3d


def get_text_targets(text: str, num_drones: int) -> np.ndarray:
    height, width = 200, 800
    image = np.zeros((height, width), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (50, 150), font, 5, 255, 10, cv2.LINE_AA)
    return get_target_points(image, num_drones)


def generate_initial_positions(num_drones: int, config: str = 'cube') -> np.ndarray:
    np.random.seed(95)  # ⚡
    if config == 'line':
        positions = np.zeros((num_drones, 3))
        positions[:, 0] = np.linspace(-10, 10, num_drones)
    elif config == 'square':
        side = int(np.ceil(np.sqrt(num_drones)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= num_drones:
                    break
                positions.append([i * 2 - side, j * 2 - side, 0])
        positions = np.array(positions[:num_drones])
    elif config == 'cube':
        side = int(np.ceil(num_drones ** (1/3)))
        positions = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(positions) >= num_drones:
                        break
                    positions.append([i * 2 - side, j * 2 - side, k * 2 - side])
        positions = np.array(positions[:num_drones])
    else:
        raise ValueError(f"Unknown configuration: {config}")
    return positions.astype(float)


def save_trajectory(trajectory: np.ndarray, filename: str):
    os.makedirs('outputs/trajectories', exist_ok=True)
    path = os.path.join('outputs/trajectories', filename)
    np.save(path, trajectory)
    print(f"Trajectory saved to {path}")


def load_trajectory(filename: str) -> np.ndarray:
    path = os.path.join('outputs/trajectories', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory {filename} not found")
    return np.load(path)