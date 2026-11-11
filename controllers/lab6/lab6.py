import numpy as np
import sys

# ============================================================
# TASK SWITCH: "pick_and_place" (lab6.wbt) or "block_stack" (lab6_extra.wbt)
# ============================================================
Task = "pick_and_place"

try:
    from controller import Supervisor
except ImportError:
    print("ERROR: Must run inside Webots as Supervisor."); sys.exit(1)


# ============================================================
# Robot Interface (DO NOT MODIFY)
# ============================================================
class UR5eInterface:
    MOTOR_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    SENSOR_NAMES = [n + "_sensor" for n in MOTOR_NAMES]
    HAND_MOTORS_1 = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]
    HAND_MOTORS_2 = ["finger_1_joint_2", "finger_2_joint_2", "finger_middle_joint_2"]

    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.motors, self.sensors = [], []
        for i in range(6):
            m = robot.getDevice(self.MOTOR_NAMES[i])
            s = robot.getDevice(self.SENSOR_NAMES[i])
            s.enable(self.timestep); m.setVelocity(1.0)
            self.motors.append(m); self.sensors.append(s)
        self.hand_motors_1, self.hand_motors_2 = [], []
        for name in self.HAND_MOTORS_1:
            m = robot.getDevice(name)
            if m: self.hand_motors_1.append(m)
        for name in self.HAND_MOTORS_2:
            m = robot.getDevice(name)
            if m: self.hand_motors_2.append(m)
        for _ in range(10): robot.step(self.timestep)

    def get_joint_positions(self):
        return np.array([s.getValue() for s in self.sensors])

    def set_joint_positions(self, q):
        for i in range(6): self.motors[i].setPosition(float(q[i]))

    def set_speed(self, speed):
        for m in self.motors: m.setVelocity(speed)

    def open_gripper(self):
        for m in self.hand_motors_1: m.setPosition(0.05)
        for m in self.hand_motors_2: m.setPosition(0.0)

    def close_gripper(self):
        for m in self.hand_motors_1: m.setPosition(0.3)
        for m in self.hand_motors_2: m.setPosition(0.8)

    def wait(self, seconds):
        for _ in range(int(seconds * 1000 / self.timestep)):
            if self.robot.step(self.timestep) == -1: return

    def move_to(self, q, wait_time=3.0):
        self.set_joint_positions(q); self.wait(wait_time)
        return self.get_joint_positions()


# ============================================================
# Constants (DO NOT MODIFY)
# ============================================================
# UR5e Standard DH Parameters
UR5E_DH_A     = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
UR5E_DH_D     = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_DH_ALPHA = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
GRIPPER_OFFSET = 0.18  # Robotiq 3F gripper length

# Webots UR5e PROTO has a 180-deg Z rotation between PROTO frame and DH frame
R_PROTO = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
ROBOT_POS = None  # Set at runtime from Supervisor

HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


# ============================================================
# PROVIDED: Coordinate Transforms (DO NOT MODIFY)
# ============================================================
def _dh(a, d, alpha, theta):
    """Standard DH transformation matrix (one joint)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]])


def world_to_base(pw):
    """Convert world position to DH base frame."""
    return R_PROTO.T @ (np.array(pw) - ROBOT_POS)


def base_to_world(pb):
    """Convert DH base frame position to world."""
    return R_PROTO @ np.array(pb) + ROBOT_POS


# ============================================================
# TASK 1: Forward Kinematics (15 pts)
# ============================================================
def forward_kinematics(q):
    """
    Compute the gripper tip position given joint angles.

    Parameters:
        q : array-like, 6 joint angles (radians)

    Returns:
        position : numpy array [x, y, z] in robot base frame
    """
    # Chain the 6 DH transforms using the provided _dh() function
    # and the UR5e DH parameters.
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(UR5E_DH_A[i], UR5E_DH_D[i], UR5E_DH_ALPHA[i], q[i])

    # Apply the tool offset (GRIPPER_OFFSET = 0.18 m) as a translation
    # along the local Z-axis of the final frame.
    T_tool = np.eye(4)
    T_tool[2, 3] = GRIPPER_OFFSET
    T = T @ T_tool

    # Return the XYZ position: first 3 elements of the last column.
    return T[:3, 3]

# For FK verification:
configs = [
    [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],   # HOME
    [0.5, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],   # shoulder pan rotated
    [0.0, -1.0,    1.0,    -1.5708, -1.5708, 0.0],   # shoulder lift + elbow changed
]

# ============================================================
# TASK 2: Numerical Jacobian (15 pts)
# ============================================================
def compute_jacobian(q, delta=1e-5):
    """
    Compute the 3x6 position Jacobian using central finite differences.

    Parameters:
        q     : numpy array (6,) current joint angles
        delta : perturbation size

    Returns:
        J : numpy array (3, 6)
    """
    q = np.array(q, dtype=float)
    J = np.zeros((3, 6))
    # for each of the 6 joints,
    for i in range(6):
        # perturb q[i] by plus or minus delta
        e_i = np.zeros(6)
        e_i[i] = delta
        # evaluate FK and compute the partial derivative
        J[:, i] = (forward_kinematics(q + e_i) - forward_kinematics(q - e_i)) / (2 * delta)
    return J


# ============================================================
# TASK 3: Define Waypoints (10 pts)
# ============================================================
def get_waypoints(block_world, bin_world):
    """
    Define the world-frame positions the gripper should visit.

    Parameters:
        block_world : [x, y, z] of block center in world frame
        bin_world   : [x, y, z] of bin center in world frame

    Returns:
        dict with keys: 'above_block', 'at_block', 'above_bin'
        Each value is a list [x, y, z] in world frame.
    """
    bx, by, bz = block_world
    gx, gy, gz = bin_world

    APPROACH = 0.15   # above the block a little
    DROP     = 0.15   # same height above the bin

    return {
        'above_block': [bx, by, bz + APPROACH],
        'at_block':    [bx, by, bz],
        'above_bin':   [gx, gy, gz + DROP],
    }


# ============================================================
# TASK 4: Gradient Descent IK (20 pts)
# ============================================================
def gradient_descent_ik(target_world, q_init, learning_rate,
                         max_iterations, tolerance, vertical=False, gripper_yaw=0.0):
    """
    Find joint angles that place the gripper at target_world.

    Parameters:
        target_world   : [x, y, z] desired position in world frame
        q_init         : starting joint angles (numpy array of 6)
        learning_rate  : step size (default 0.5)
        max_iterations : max steps (default 1000)
        tolerance      : converge when error < this (meters)
        vertical       : stacking only; keep the gripper pointing down with fixed yaw
        gripper_yaw    : rotation about world vertical (radians), when vertical=True

    Returns:
        q_solution : numpy array of 6 joint angles
        converged  : bool
        errors     : list of position error at each iteration
    """

    target_base = world_to_base(target_world)
    q = np.array(q_init, dtype=float)
    if vertical:
        # The last wrist joint adds yaw without changing the downward tool axis
        # or tip position. Its sign is reversed because the tool points down.
        # Only q0, q1, q2 are free; the other joints maintain this tool pose.
        joint_map = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                              [0, -1, -1], [0, 0, 0], [1, 0, 0]])
        q[3] = -np.pi / 2 - q[1] - q[2]
        q[4] = -np.pi / 2
        q[5] = q[0] - gripper_yaw
    errors = []
    for i in range(max_iterations):
        current_pos = forward_kinematics(q)
        error = target_base - current_pos
        error_distance = np.linalg.norm(error)
        errors.append(error_distance)
        if error_distance < tolerance:
            return q, True, errors
        jacobian = compute_jacobian(q)
        if vertical:
            # Chain rule: account for the wrist compensating for shoulder/elbow
            # motion. Setting wrist angles AFTER ordinary IK would miss the target.
            update = joint_map @ (learning_rate * (jacobian @ joint_map).T @ error)
            scale = 1.0
            for angle, change in zip(q, update):
                if change > 0:
                    scale = min(scale, (2 * np.pi - angle) / change)
                elif change < 0:
                    scale = min(scale, (-2 * np.pi - angle) / change)
            q += max(0.0, scale) * update
        else:
            q = q + learning_rate * jacobian.T @ error
            q = np.clip(q, -np.pi * 2, np.pi * 2)
    return q, False, errors


# ============================================================
# TASK 5: Pick and Place Sequence (10 pts)
# ============================================================
def pick_and_place(arm, block_world, bin_world):
    """
    Pick up the block and place it in the bin.

    """
    waypoints = get_waypoints(block_world, bin_world)
    solutions = {}
    q_start = np.array(HOME)
    for name, position in waypoints.items():
        q, converged, errors = gradient_descent_ik(position, q_start, 0.5, 1000, 0.005)
        if not converged:
            raise RuntimeError(f"IK failed for {name}: final error {errors[-1]:.4f} m")
        solutions[name] = q
        q_start = q
    # Now the flow
    arm.open_gripper()
    arm.move_to(solutions["above_block"], 3.0)
    arm.move_to(solutions["at_block"], 2.0)
    arm.close_gripper()
    arm.wait(1.0)
    arm.move_to(solutions["above_block"], 2.0)
    arm.move_to(solutions["above_bin"], 3.0)
    arm.open_gripper()
    arm.wait(20.0)




# ============================================================
# EXTRA CREDIT: Block Stacking
# ============================================================
BLOCK_SIZE = 0.05
STACK_GRASP_OFFSET = 0.015  # grip 15 mm above center, 10 mm below the block's top
STACK_GRIPPER_YAW = np.deg2rad(45.0)  # diagonal grip: paired fingers straddle a corner
STACK_APPROACH_CLEARANCE = 0.08  # 15 cm above the fourth block exceeds vertical-pose reach
# Stack order from bottom to top. Gray is placed first (bottom),
# so the finished stack reads red -> blue -> green -> gray from the top down.
STACK_ORDER = ["GRAY_BLOCK", "GREEN_BLOCK", "BLUE_BLOCK", "RED_BLOCK"]


def _stack_move_to(arm, q, wait_time):
    """Synchronize joint arrival so the wrist compensates throughout a move."""
    delta = np.abs(np.array(q) - arm.get_joint_positions())
    duration = max(wait_time, float(np.max(delta)) / 0.7)
    # Equal joint speeds let small wrist rotations finish before larger shoulder
    # rotations. Proportional speeds keep the downward pose during travel too.
    for motor, change in zip(arm.motors, delta):
        motor.setVelocity(max(float(change) / duration, 0.001))
    actual = arm.move_to(q, duration + 0.2)
    arm.set_speed(1.0)
    return actual


def _stack_vertical_move(arm, target, label):
    """Lower/lift along a Cartesian line instead of a curved joint-space path."""
    q = arm.get_joint_positions()
    start = base_to_world(forward_kinematics(q))
    target = np.array(target, dtype=float)
    steps = max(1, int(np.ceil(np.linalg.norm(target - start) / 0.01)))
    for step in range(1, steps + 1):
        position = start + (target - start) * step / steps
        q, converged, errors = gradient_descent_ik(position, q, 0.5, 8000, 0.001,
                                                  vertical=True, gripper_yaw=STACK_GRIPPER_YAW)
        if not converged:
            raise RuntimeError(f"IK failed for {label}: final error {errors[-1]:.4f} m")
        q = _stack_move_to(arm, q, 0.15)


def block_stack(arm, robot):
    """
    Stack the four blocks on the STACK_ZONE using the same IK solver
    and waypoint helper as pick_and_place().
    """
    zone = np.array(robot.getFromDef("STACK_ZONE").getPosition())
    q_start = np.array(HOME)

    for level, name in enumerate(STACK_ORDER):
        block_world = robot.getFromDef(name).getPosition()
        # Block-center height this block should end up at in the stack
        place_world = [zone[0], zone[1], zone[2] + BLOCK_SIZE / 2 + level * BLOCK_SIZE + 0.005]

        # Grip near the top; retain the same tip-to-block height when placing.
        grasp_world = np.array(block_world) + [0, 0, STACK_GRASP_OFFSET]
        place_tip = np.array(place_world) + [0, 0, STACK_GRASP_OFFSET]
        # Reuse get_waypoints: 'above_bin' is the position above the stack
        waypoints = get_waypoints(grasp_world, place_tip)
        waypoints['above_bin'] = place_tip + [0, 0, STACK_APPROACH_CLEARANCE]
        waypoints['at_place'] = place_tip

        solutions = {}
        for key in ['above_block', 'at_block', 'above_bin', 'at_place']:
            q, converged, errors = gradient_descent_ik(waypoints[key], q_start, 0.5,
                                                      8000, 0.001, vertical=True,
                                                      gripper_yaw=STACK_GRIPPER_YAW)
            if not converged:
                raise RuntimeError(f"IK failed for {name} {key}: final error {errors[-1]:.4f} m")
            solutions[key] = q
            q_start = q

        # Pick
        arm.open_gripper()
        _stack_move_to(arm, solutions["above_block"], 3.0)
        _stack_vertical_move(arm, waypoints["at_block"], f"{name} lower to pickup")
        arm.close_gripper()
        arm.wait(1.0)
        _stack_vertical_move(arm, waypoints["above_block"], f"{name} lift")
        # Place on the stack
        _stack_move_to(arm, solutions["above_bin"], 3.0)
        _stack_vertical_move(arm, waypoints["at_place"], f"{name} lower onto stack")
        arm.open_gripper()
        arm.wait(1.0)
        _stack_vertical_move(arm, waypoints["above_bin"], f"{name} retreat")
        # Start the next block from the actual retreat pose, not at_place.
        q_start = arm.get_joint_positions()

    arm.move_to(np.array(HOME), 3.0)
    arm.wait(5.0)


# ============================================================
# MAIN — DO NOT MODIFY
# ============================================================
def main():
    global ROBOT_POS
    robot = Supervisor()
    arm = UR5eInterface(robot)
    ROBOT_POS = np.array(robot.getSelf().getPosition())
    arm.wait(1.0) # Wait for block to fall
    if Task == "block_stack":
        block_stack(arm, robot)
        return
    block_world = robot.getFromDef("BLOCK").getPosition()
    bin_world   = robot.getFromDef("BIN").getPosition()
    pick_and_place(arm, block_world, bin_world)


if __name__ == "__main__":
    main()
