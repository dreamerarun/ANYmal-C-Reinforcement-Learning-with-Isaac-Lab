# ANYmal C Reinforcement Learning with Isaac Lab

Reinforcement learning of the **ANYmal C quadruped robot** using **NVIDIA Isaac Lab**, **Isaac Sim**, **RSL-RL**, and **Proximal Policy Optimization (PPO)**.

The project explores learned quadrupedal locomotion in two simulation environments:

* Flat terrain
* Procedurally generated rough terrain

The trained policies are exported to both **PyTorch (`.pt`)** and **ONNX (`.onnx`)** formats for subsequent evaluation and deployment experiments.

---

## Project Overview

The objective of this project is to train an ANYmal C quadruped to learn velocity-based locomotion using reinforcement learning.

The agent observes the simulated robot state and terrain information, produces joint-position actions, and receives rewards based on locomotion performance, stability, energy-related penalties, and contact behavior.

### RL Pipeline

```text
                 ┌──────────────────────┐
                 │   Isaac Sim /        │
                 │     Isaac Lab        │
                 └──────────┬───────────┘
                            │
                            ▼
                    Robot Observation
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Actor-Critic       │
                 │   Policy Network     │
                 └──────────┬───────────┘
                            │
                            ▼
                     Joint Actions
                            │
                            ▼
                 ┌──────────────────────┐
                 │      ANYmal C        │
                 │   Quadruped Robot    │
                 └──────────┬───────────┘
                            │
                            ▼
                         Reward
                            │
                            ▼
                 ┌──────────────────────┐
                 │       PPO            │
                 │   Policy Update      │
                 └──────────┬───────────┘
                            │
                            └──────────────► Repeat
```

---

## Technologies

* NVIDIA Isaac Lab
* NVIDIA Isaac Sim
* RSL-RL
* Proximal Policy Optimization (PPO)
* PyTorch
* ONNX
* CUDA
* Python
* PhysX

---

## Robot

The project uses the **ANYmal C** quadruped robot from ANYbotics.

The robot has four legs with three actuated joints per leg:

```text
                ANYmal C
                   │
        ┌──────────┴──────────┐
        │                     │
     Front Legs            Rear Legs
        │                     │
     HAA/HFE/KFE          HAA/HFE/KFE
```

The policy controls the robot through joint-position actions.

---

# Experiments

## 1. Flat Terrain

The first experiment trains ANYmal C on a flat environment.

### Configuration

| Parameter             |        Value |
| --------------------- | -----------: |
| Algorithm             |          PPO |
| Policy                | Actor-Critic |
| Seed                  |           42 |
| Parallel environments |          500 |
| Steps per environment |           24 |
| Training iterations   |        1,500 |
| Approx. transitions   |   18 million |
| Learning rate         |        0.001 |
| Gamma                 |         0.99 |
| GAE lambda            |         0.95 |
| Learning epochs       |            5 |
| Mini-batches          |            2 |
| Entropy coefficient   |         0.01 |
| Clip parameter        |          0.2 |
| Activation            |          ELU |

### Network

```text
Actor:
128 → 128 → 128

Critic:
128 → 128 → 128
```

The experiment uses 500 parallel environments to increase simulation throughput.

---

## 2. Rough Terrain

The second experiment extends the locomotion task to procedurally generated terrain.

### Configuration

| Parameter             |        Value |
| --------------------- | -----------: |
| Algorithm             |          PPO |
| Policy                | Actor-Critic |
| Seed                  |           42 |
| Parallel environments |           25 |
| Steps per environment |           32 |
| Training iterations   |        3,000 |
| Approx. transitions   |  2.4 million |
| Learning rate         |        0.001 |
| Gamma                 |         0.99 |
| GAE lambda            |         0.95 |
| Learning epochs       |            4 |
| Mini-batches          |            2 |
| Entropy coefficient   |         0.01 |
| Clip parameter        |          0.2 |
| Activation            |          ELU |

### Network

```text
Actor:
512 → 256 → 128

Critic:
512 → 256 → 128
```

The rough-terrain policy uses a larger neural network than the flat-terrain experiment.

---

# Rough Terrain Generation

The rough-terrain environment contains multiple procedurally generated terrain types.

Examples include:

* Pyramid stairs
* Inverted pyramid stairs
* Box/grid terrain
* Random rough terrain
* Sloped terrain
* Inverted sloped terrain

```text
                 Rough Terrain
                       │
       ┌───────────────┼────────────────┐
       │               │                │
    Stairs          Rough            Slopes
       │            Terrain             │
       │               │                │
   Inverted         Height           Inverted
    Stairs          Field             Slopes
```

The environment uses a height scanner to provide terrain information to the policy.

---

# Observations

The policy observation space includes information such as:

* Base linear velocity
* Base angular velocity
* Projected gravity
* Velocity commands
* Relative joint positions
* Relative joint velocities
* Previous actions
* Terrain height scan

The rough-terrain configuration additionally uses terrain sensing through a ray-caster height scanner.

---

# Actions

The policy produces joint-position actions for the robot.

```text
Policy
   │
   ▼
Joint Position Actions
   │
   ▼
ANYmal C Actuators
   │
   ▼
Robot Motion
```

The action scale is configured as:

```text
0.25
```

---

# Reward Design

The locomotion task combines multiple reward and penalty terms.

Important terms include:

### Velocity Tracking

The robot is rewarded for following commanded linear and angular velocities.

```text
Linear velocity tracking
Angular velocity tracking
```

### Stability

Penalties are applied for undesirable motion such as:

```text
Vertical linear velocity
Roll/pitch angular velocity
Non-flat orientation
```

### Energy and Smoothness

Additional penalties discourage excessive actuator effort and rapidly changing actions:

```text
Joint torque penalty
Joint acceleration penalty
Action-rate penalty
```

### Foot Motion

The reward also considers foot air-time behavior.

### Undesired Contacts

Unwanted contacts, such as thigh contacts with the environment, are penalized.

---

# Domain Randomization

The rough-terrain experiment introduces environment variation to improve robustness.

Examples include:

* Physics material randomization
* Base mass randomization
* Initial pose randomization
* Initial velocity randomization
* Joint reset randomization
* Periodic robot pushes

The robot can therefore experience disturbances during training instead of learning only a single fixed simulation condition.

---

# PPO

The project uses Proximal Policy Optimization through RSL-RL.

The main PPO parameters are:

```text
Learning rate       = 0.001
Gamma               = 0.99
GAE lambda          = 0.95
Clip parameter      = 0.2
Entropy coefficient = 0.01
Desired KL          = 0.01
Max gradient norm   = 1.0
```

The optimizer uses adaptive learning-rate scheduling based on the configured KL target.

---

# Trained Policies

The exported policies are provided in two formats.

## PyTorch

```text
policies/
├── flat/
│   └── policy.pt
│
└── rough/
    └── policy.pt
```

## ONNX

```text
policies/
├── flat/
│   └── policy.onnx
│
└── rough/
    └── policy.onnx
```

ONNX export is included to facilitate future inference and deployment experiments outside the original training pipeline.

---

# Repository Structure

```text
anymal-rl-isaaclab/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── flat/
│   │   ├── agent.yaml
│   │   └── env.yaml
│   │
│   └── rough/
│       ├── agent.yaml
│       └── env.yaml
│
├── policies/
│   ├── flat/
│   │   ├── policy.onnx
│   │   └── policy.pt
│   │
│   └── rough/
│       ├── policy.onnx
│       └── policy.pt
│
├── results/
│   ├── flat/
│   └── rough/
│
├── docs/
│   └── diffs/
│       ├── flat-IsaacLab.diff
│       └── rough-IsaacLab.diff
│
└── scripts/
```

---

# Reproducing the Experiment

## Prerequisites

Recommended environment:

* Linux
* NVIDIA GPU
* NVIDIA GPU driver compatible with the installed CUDA/Isaac Sim version
* NVIDIA Isaac Sim
* NVIDIA Isaac Lab
* Python environment configured for Isaac Lab
* RSL-RL
* PyTorch

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/anymal-rl-isaaclab.git
cd anymal-rl-isaaclab
```

The repository intentionally does not contain the complete Isaac Lab installation.

Install Isaac Lab separately according to the official NVIDIA Isaac Lab installation instructions.

---

# Training

The configuration snapshots used for the experiments are available under:

```text
configs/flat/
configs/rough/
```

The exact original training runs are preserved through the exported policies and configuration snapshots.

---

# Evaluation

The exported ONNX policies can be used for subsequent inference and deployment experiments.

Example workflow:

```text
Trained PPO Policy
        │
        ▼
     Export
        │
   ┌────┴────┐
   ▼         ▼
 PyTorch    ONNX
   │         │
   └────┬────┘
        ▼
   Inference
        │
        ▼
    ANYmal C
```

---

# Results

Training artifacts are organized by environment:

```text
results/
├── flat/
└── rough/
```

Training curves and evaluation videos can be added here.

Recommended future additions:

* Episode reward curves
* Episode length curves
* Policy loss
* Value loss
* Learning rate
* PPO KL divergence
* Evaluation videos
* Flat vs. rough terrain comparison

---

# Key Observations

The two experiments demonstrate two different levels of locomotion complexity.

### Flat Terrain

The flat-terrain experiment focuses on learning basic velocity tracking and stable quadrupedal locomotion using a relatively compact policy network.

### Rough Terrain

The rough-terrain experiment introduces:

* Procedural terrain
* Terrain height observations
* Contact sensing
* Domain randomization
* External disturbances
* A larger actor-critic network

This provides a more challenging setting for learning robust locomotion.

---

# Future Work

Potential extensions include:

* Longer training runs
* Improved reward tuning
* Curriculum learning
* More aggressive terrain randomization
* Larger terrain difficulty ranges
* Push-recovery evaluation
* Zero-shot terrain transfer
* Sim-to-real experiments
* Real-time policy inference
* ROS 2 integration
* Deployment on NVIDIA Jetson
* Comparison with other RL algorithms
* Domain randomization for sim-to-real transfer

---

# References

* NVIDIA Isaac Lab
* NVIDIA Isaac Sim
* RSL-RL
* PPO
* ANYbotics ANYmal

---

# Author

**Arun M.**

Robotics & Automation Engineer
Interested in Reinforcement Learning, Physical AI, Robot Learning, ROS 2, NVIDIA Isaac Sim, Isaac Lab, and Autonomous Robotics.

---

## Disclaimer

This repository contains experiment configurations, trained policy artifacts, and documentation from reinforcement-learning experiments performed in simulation.

The complete Isaac Lab/Isaac Sim installation and NVIDIA-provided assets are not redistributed in this repository.
