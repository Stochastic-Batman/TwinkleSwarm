# TwinkleSwarm
Illuminated Drone Show Simulation

TwinkleSwarm is a 3D drone swarm simulation project. It models coordinated motion of multiple drones to create illuminated formations and animations.

The swarm can:

* Form static shapes from images or text
* Transition between formations
* Track motion from a video while preserving the swarm shape
* Visualize the result in 3D and export it as a video file

The project focuses on numerical simulation and trajectory generation, not real drone control.

## Features

* 3D drone swarm simulation
* Static formation from handwritten images
* Text-based formations
* Smooth transition between shapes
* Motion tracking from video input
* Shape preservation during motion
* Collision avoidance between drones
* Velocity-limited motion
* 3D visualization
* MP4 video export

## Project Structure

```
├── data/
│   ├── images/  # Handwritten images or masks
│   └── videos/  # Input videos for motion tracking
├── src/
│   ├── drone_dynamics.py  # Motion and control models
│   ├── utils.py  # Utility functions
│   ├── video_processing.py  # Optical flow extraction
│   └── visualize.py  # 3D animation - by Claude Sonnet 4.5
├── outputs/
│   ├── trajectories/  # Saved drone trajectories
│   └── videos/  # Exported MP4 files
├── main.py
├── examples.py  # by Claude Sonnet 4.5
├── requirements.txt  # you might need to add ffmpeg to PATH on Windows  
├── TwinkleSwarm.tex  # source code for documentation
├── TwinkleSwarm.pdf  # documentation
└── README.md
```

## Simulation Model (High-Level)

Each drone is described by:

* A 3D position
* A 3D velocity

The swarm evolves over time using:

* Velocity tracking toward targets or a velocity field
* Repulsion to avoid collisions
* Damping for smooth convergence
* A maximum velocity limit

For dynamic scenes, motion is extracted from a video and converted into a velocity field that drives the swarm.

## Supported Inputs

* Handwritten image files (PNG, JPG)
* Text strings for shape generation
* Video files for motion tracking
* Configurable number of drones
* Custom or automatic initial positions

## Outputs

The simulation produces:

1. Drone trajectories (NumPy arrays saved as .npy files)
2. A 3D animated visualization (displayed during runtime)
3. A saved MP4 video of the animation

Output files are stored in the `outputs/` directory.

## Setting Up & Running The Project

Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

```bash
python -m venv twinkleswarm_venv
```

Activate the virtual environment:
* For Linux and macOS: `source twinkleswarm_venv/bin/activate`
* For Windows (from CMD): `twinkleswarm_venv\Scripts\activate.bat`


Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare your input data:
* Place your handwritten name image in `data/images/`. Example: `data/images/handwritten_name.jpg`
* Place your video in `data/videos/`. Example: `data/videos/wrecking_ball.mp4`

and run the main entry point:

```bash
python main.py
```

This will:

* Generate the swarm trajectories
* Display a 3D animation
* Save the animation as an MP4 file

## Limitations

* Simulation only, no real drone control
* Performance depends on the number of drones
* Dense formations may require tuning
* Motion tracking quality depends on video quality
* Add a short **“Quick Demo”** section
* Write a **configuration example** section

## Troubleshooting
Problem: Drones don't converge
* Increase `k_d` (damping) or decrease `k_p` (attraction)
* Increase `T_final` (simulation time)

Problem: Drones collide
* Increase `k_rep` (repulsion strength)
* Increase `r_safe` (safety radius)
Decrease num_drones or spread targets further

Problem: Motion too slow
* Increase `v_max` (maximum velocity)
* Increase `k_p` (attraction) or `k_v` (velocity tracking)

Problem: Oscillations around target
* Increase `k_d` (damping)
* Ensure `k_d >= 2 * sqrt(m * k_p)` for critical damping

Problem: Video tracking doesn't work
* Check video quality (needs clear motion, good contrast)
* Adjust `video_scale` and `blur_sigma` in config
* Try different optical flow parameters