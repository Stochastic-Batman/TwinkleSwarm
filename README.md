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
│   ├── visualize.py  # 3D animation
│   └── utils.py  # Utility functions
├── outputs/
│   ├── trajectories/  # Saved drone trajectories
│   └── videos/  # Exported MP4 files
├── main.py
├── requirements.txt
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

1. Drone trajectories
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
