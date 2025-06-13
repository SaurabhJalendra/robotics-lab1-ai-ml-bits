import pybullet as p
import pybullet_data
import numpy as np
import time
import sys

# --- Create a simple box robot ---
def create_box_robot(position=[0, 0, 0.1], half_extents=[0.2, 0.2, 0.1]):
    try:
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])
        robot_id = p.createMultiBody(baseMass=1,
                                     baseCollisionShapeIndex=collision_shape,
                                     baseVisualShapeIndex=visual_shape,
                                     basePosition=position)
        return robot_id
    except Exception as e:
        print(f"Error creating box robot: {e}")
        return -1

# --- Create semi-circular obstacles ---
def create_obstacles(radius=5.0, count=18, height=2.0):
    try:
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
        half_extents = [0.15, 0.15, height / 2.0]
        for i in range(count):
            angle = np.pi * i / (count - 1)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            color = colors[i % len(colors)]
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=col_shape,
                              baseVisualShapeIndex=vis_shape,
                              basePosition=[x, y, half_extents[2]])
    except Exception as e:
        print(f"Error creating obstacles: {e}")

# --- Move the robot (forward or turn) ---
def move_robot(robot_id, velocity=0.15, turn=False):
    try:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        if turn:
            yaw += np.pi / 3  # 60 degree turn
        dx = velocity * np.cos(yaw)
        dy = velocity * np.sin(yaw)
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
        new_orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(robot_id, new_pos, new_orn)
    except Exception as e:
        print(f"Error moving robot: {e}")

# --- Cast 5 beams at angles [-40°, -20°, 0°, 20°, 40°] ---
def cast_multiple_beams(robot_id, scan_range=8.0, noise_std=0.02):
    try:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        beam_angles_deg = [-40, -20, 0, 20, 40]
        beam_data = []

        for angle_deg in beam_angles_deg:
            angle_rad = yaw + np.deg2rad(angle_deg)
            beam_dir = [np.cos(angle_rad), np.sin(angle_rad), 0]
            beam_end = [pos[0] + scan_range * beam_dir[0],
                        pos[1] + scan_range * beam_dir[1],
                        pos[2]]

            ray_result = p.rayTest(pos, beam_end)[0]
            hit_fraction = ray_result[2]
            distance = hit_fraction * scan_range

            noisy_distance = distance + np.random.normal(0, noise_std)
            noisy_distance = np.clip(noisy_distance, 0.0, scan_range)

            beam_data.append((noisy_distance, beam_dir, beam_end))

        return pos, beam_data
    except Exception as e:
        print(f"Error casting beams: {e}")
        return [0, 0, 0], []

# --- Main Simulation ---
try:
    physicsClient = p.connect(p.GUI)
    if physicsClient < 0:
        print("CRITICAL ERROR: Could not connect to PyBullet.")
        sys.exit(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    planeId = p.loadURDF("plane.urdf")
    if planeId < 0:
        print("CRITICAL ERROR: Could not load plane.")
        if p.isConnected(): p.disconnect()
        sys.exit(1)

    robot_id = create_box_robot()
    if robot_id < 0:
        print("CRITICAL ERROR: Could not create robot.")
        if p.isConnected(): p.disconnect()
        sys.exit(1)

    create_obstacles()

    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )

    # Sensor and control parameters
    scan_range = 8.0
    stop_distance = 0.2
    sensor_noise_std_dev = 0.02

    while True:
        p.stepSimulation()
        p.removeAllUserDebugItems()

        # Cast 5 beams
        pos, beam_data = cast_multiple_beams(robot_id, scan_range, sensor_noise_std_dev)

        min_distance = scan_range

        # Visualize beams + text
        for distance, beam_dir, beam_end in beam_data:
            color = [1, 0, 0] if distance < stop_distance else [0, 1, 0]
            p.addUserDebugLine(pos, beam_end, color, lineWidth=2, lifeTime=0.1)

            label_pos = [pos[0] + 0.2 * beam_dir[0],
                         pos[1] + 0.2 * beam_dir[1],
                         pos[2] + 0.1]
            p.addUserDebugText(f"{distance:.2f} m", label_pos,
                               textColorRGB=[0, 0, 0],
                               textSize=0.7,
                               lifeTime=0.1)

            min_distance = min(min_distance, distance)

        # Movement decision
        if min_distance < stop_distance:
            move_robot(robot_id, velocity=0.15, turn=True)
        else:
            move_robot(robot_id, velocity=0.15)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except p.error as e:
    print(f"PyBullet Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
finally:
    if p.isConnected():
        p.disconnect()
        print("PyBullet disconnected cleanly.")
    else:
        print("PyBullet was not connected or already disconnected.")
