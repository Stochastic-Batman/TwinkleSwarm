import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


np.random.seed(95)  # ⚡


class OpticalFlowProcessor:
    def __init__(self, video_path: str, scale: float = 0.01, blur_sigma: float = 2.0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = scale
        self.blur_sigma = blur_sigma
        self.flows = []
        self.prev_gray = None

    def compute_optical_flow(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_smoothed = np.zeros_like(flow)
                flow_smoothed[:, :, 0] = gaussian_filter(flow[:, :, 0], sigma=self.blur_sigma)
                flow_smoothed[:, :, 1] = gaussian_filter(flow[:, :, 1], sigma=self.blur_sigma)
                self.flows.append(flow_smoothed)
            self.prev_gray = gray
            frame_idx += 1
        self.cap.release()
        return np.array(self.flows)

    def get_velocity_field_function(self, flows: np.ndarray):
        def velocity_field(position: np.ndarray, time: float) -> np.ndarray:
            frame_idx = int(time * self.fps)
            if frame_idx < 0 or frame_idx >= len(flows):
                return np.zeros(3)
            x_pixel = position[0] / self.scale + self.width / 2
            y_pixel = self.height / 2 - position[1] / self.scale
            x_pixel = np.clip(x_pixel, 0, self.width - 1.001)
            y_pixel = np.clip(y_pixel, 0, self.height - 1.001)
            x0, y0 = int(x_pixel), int(y_pixel)
            x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)
            alpha = x_pixel - x0
            beta = y_pixel - y0
            flow = flows[frame_idx]
            v00 = flow[y0, x0]
            v10 = flow[y0, x1]
            v01 = flow[y1, x0]
            v11 = flow[y1, x1]
            v_interp = (1 - alpha) * (1 - beta) * v00 + alpha * (1 - beta) * v10 + (1 - alpha) * beta * v01 + alpha * beta * v11
            v_physical = np.array([v_interp[0] * self.scale * self.fps, -v_interp[1] * self.scale * self.fps, 0.0])
            return v_physical
        return velocity_field

def extract_shape_from_video(video_path: str, num_drones: int, frame_index: int = 0) -> np.ndarray:
    np.random.seed(95)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area = np.sum(binary == 255) / binary.size
    if area > 0.5:
        binary = 255 - binary
    points = np.argwhere(binary == 255)
    points = points[:, [1, 0]]
    if len(points) == 0:
        raise ValueError("No points found in video frame")
    replace = len(points) < num_drones
    indices = np.random.choice(len(points), num_drones, replace=replace)
    targets_2d = points[indices]
    height, width = gray.shape
    scale = 0.01
    targets_2d_centered = targets_2d - np.array([width / 2, height / 2])
    targets_3d = np.zeros((num_drones, 3))
    targets_3d[:, 0] = targets_2d_centered[:, 0] * scale
    targets_3d[:, 1] = -targets_2d_centered[:, 1] * scale
    targets_3d -= np.mean(targets_3d, axis=0)
    return targets_3d