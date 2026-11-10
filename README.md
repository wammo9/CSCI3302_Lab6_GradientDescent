# CSCI 3302 Introduction to Robotics
## Lab 6: Gradient Descent Inverse Kinematics for Pick and Place

**Due:** November 12th @ 11:59 PM
**Submission:** Canvas — `lab6.py` + report
**Points:** 100 + 10 extra credit

**Lab Report:** [Lab 6 Report (Google Doc)](https://docs.google.com/document/d/18okTkoGSIDv2MXWVTl13_SYpBz2plMt64-YRmdzHGR0/edit?usp=sharing)

## Overview

A UR5e robot arm with a Robotiq 3F gripper picks up a red block and places it in the green bin (Webots). We implement forward kinematics, a numerical Jacobian, and gradient descent inverse kinematics to compute the joint angles needed to reach target positions in 3D space. The gripper interface is provided; our work is the kinematics and task sequencing.

## Background

- **Forward kinematics (DH convention):** chain 6 Denavit-Hartenberg transforms (one per UR5e joint) using the provided `_dh(a, d, alpha, theta)` function to get the end-effector position.
- **Numerical Jacobian:** 3×6 position Jacobian via central finite differences, `J[:, i] = (FK(q + δe_i) − FK(q − δe_i)) / (2δ)`, with δ = 1e-5.
- **Gradient descent IK (Jacobian transpose):** `q ← q + α · Jᵀ · (target − FK(q))`, iterated until the position error is below tolerance (5 mm).

## Tasks

| # | Task | Points |
|---|------|--------|
| 1 | **Forward Kinematics** — `forward_kinematics(q)` returns gripper tip [x, y, z] in the base frame. Chain the 6 DH transforms, apply `GRIPPER_OFFSET` (0.18 m) along local Z, return the first 3 elements of the last column. | 15 |
| 2 | **Numerical Jacobian** — `compute_jacobian(q)` returns the 3×6 Jacobian via central finite differences using FK from Task 1. | 15 |
| 3 | **Pick-and-Place Waypoints** — `get_waypoints(block_world, bin_world)` returns a dict of 3D world-frame targets (approach height, grasp position, lift height, etc.). | 10 |
| 4 | **Gradient Descent IK** — `gradient_descent_ik()`: convert target with `world_to_base()`, compute error, compute Jacobian, update `q = q + learning_rate · Jᵀ @ error`, clip joints to [−2π, 2π], repeat until `‖error‖ < tolerance`. | 20 |
| 5 | **Pick-and-Place Sequence** — `pick_and_place()` uses IK for each waypoint and commands the arm and gripper through the full sequence. | 10 |
| — | **Report** (2–3 pages) | 30 |
| — | **Extra credit** — block stacking | 10 |

## Report (30 pts)

Link: [Lab 6 Report](https://docs.google.com/document/d/18okTkoGSIDv2MXWVTl13_SYpBz2plMt64-YRmdzHGR0/edit?usp=sharing)

1. **Verification (9 pts):** Test FK at 3 joint configurations; report joint angles, FK output, and world position. Verify at least one visually in Webots. Explain what `GRIPPER_OFFSET` does and why it matters.
2. **IK Analysis (9 pts):** Run IK on 3 target positions; report iterations to converge, final error, and joint angles. Compare learning rates 0.1 vs 0.5 vs 2.0 and plot error vs. iteration for one target.
3. **Reflection (9 pts):** Role of the Jacobian in IK, why Jᵀ instead of J⁻¹, and limitations (orientation, singularities, local minima). Include a screenshot of the completed pick-and-place.
4. **Team Contributions (3 pts):** What each team member contributed.

## API Reference

| Method | Description |
|--------|-------------|
| `arm.move_to(q, wait_time)` | Move to joint angles, wait to arrive |
| `arm.open_gripper()` | Open the Robotiq 3F gripper |
| `arm.close_gripper()` | Close gripper (fingertip pinch) |
| `arm.wait(seconds)` | Wait for simulation time |
| `arm.set_speed(speed)` | Set joint velocity (rad/s) |
| `_dh(a, d, alpha, theta)` | Provided: 4×4 DH transform matrix |
| `world_to_base(pos)` / `base_to_world(pos)` | Provided: convert between world and base frames |
| `forward_kinematics(q)` | YOUR IMPL: gripper [x,y,z] in base frame |
| `compute_jacobian(q)` | YOUR IMPL: 3×6 position Jacobian |
| `gradient_descent_ik(target, q0)` | YOUR IMPL: solve IK via Jᵀ method |

## Extra Credit (10 pts)

Using the new world file (`lab_6_extra.wbt`) with four blocks (gray, red, blue, green), use the same IK solver to stack the blocks. Stacking order: red → blue → green → gray (first).

The task is selected by a single variable:

```python
Task = "block_stack"   # or "pick_and_place"
```
