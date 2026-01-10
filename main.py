import sys
import numpy as np
from src.utils.load_data import load_image, load_video


np.random.seed(95)  # ⚡

def main():
    try:
        img = load_image('handwritten_name.jpg')
        greeting = "Happy New Year!"
        vid = load_video('wrecking_ball.mp4')
        print("Inputs loaded successfully.")
        initial_positions = np.random.rand(100, 3) * 10
        num_drones = initial_positions.shape[0]
        # sub-problem 1 targets
        # name_targets = get_name_targets(img, num_drones)
        # sub-problem 2 targets
        # greeting_targets = get_text_targets(greeting, num_drones)
        # sub-problem 3 motion
        # velocity_field = extract_velocity_field(vid)
        vid.release()
    except Exception as e:
        print(f"Error loading inputs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()