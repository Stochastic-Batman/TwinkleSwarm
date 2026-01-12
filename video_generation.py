import cv2
import numpy as np


def generate_purple_ball_video(video_path: str, width: int=640, height: int=480, fps: int=30, duration: float=5.0, radius: int=50, amp: int=200, freq: float=0.5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    center_y = height // 2
    purple = (128, 0, 128)
    black = (0, 0, 0)
    for frame in range(int(duration * fps)):
        t = frame / fps
        center_x = width // 2 + int(amp * np.sin(2 * np.pi * freq * t))
        img = np.full((height, width, 3), black, dtype=np.uint8)
        cv2.circle(img, (center_x, center_y), radius, purple, -1)
        out.write(img)
    out.release()
    print(f"Purple ball video generated at {video_path}")


if __name__ == "__main__":
    generate_purple_ball_video("data/videos/purple_ball.mp4")