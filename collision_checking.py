import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from generating_environment import scene_from_file, Environment
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collision checking with obstacles.")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Defines the robot to use: 'arm' or 'freeBody'.")
    parser.add_argument("--map", type=str, required=True, help="Filename of the map containing obstacles.")
    return parser.parse_args()

def environment_to_array(environment: Environment) -> np.ndarray:
    # Create an empty list to store the obstacle data
    obstacles_array = []

    # Iterate over the obstacles in the environment
    for obstacle in environment.obstacles:
        # Extract x (center_x), y (center_y), width, height, and pose
        x = obstacle.center[0]
        y = obstacle.center[1]
        width = obstacle.width
        height = obstacle.height
        pose = obstacle.pose
        
        # Append the obstacle data as a tuple
        obstacles_array.append([x, y, width, height, pose])
    
    # Convert the list to a NumPy array
    return np.array(obstacles_array)

def h_transform(edges, x, y, pose):
    transform = np.array([
        [np.cos(pose), -np.sin(pose), x],
        [np.sin(pose),  np.cos(pose), y],
        [0,             0,            1]
    ])
    t_edges = np.hstack([edges, np.ones((edges.shape[0], 1))])
    transformed_edges = t_edges @ transform.T
    return transformed_edges[:, :2]

def get_axes(vertices):
    axes = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % num_vertices]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        length = np.linalg.norm(normal)
        if length != 0:
            normal = normal / length
        axes.append(normal)
    return axes

def project(vertices, axis):
    projections = vertices @ axis
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    return min_proj, max_proj

def is_separating_axis(proj1, proj2):
    return proj1[1] < proj2[0] or proj2[1] < proj1[0]

def polygons_collide(vertices1, vertices2):
    axes1 = get_axes(vertices1)
    axes2 = get_axes(vertices2)
    axes = axes1 + axes2
    for axis in axes:
        proj1 = project(vertices1, axis)
        proj2 = project(vertices2, axis)
        if is_separating_axis(proj1, proj2):
            return False
    return True

def arm_forward_kinematics(joint_angles, base_position):
    # Given two joint angles (in radians), calculate the position of each joint and the end-effector
    l1, l2 = 2, 1.5  # lengths of link 1 and link 2
    theta1, theta2 = joint_angles
    base_x, base_y = base_position
    # Position of first joint
    x1 = base_x + l1 * np.cos(theta1)
    y1 = base_y + l1 * np.sin(theta1)
    # Position of second joint (end effector)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([[base_x, base_y], [x1, y1], [x2, y2]])

def freebody_random_pose():
    """Generate a random pose for a free body."""
    x = np.random.uniform(0, 20)
    y = np.random.uniform(0, 20)
    pose = np.random.uniform(0, 2 * np.pi)  # Random rotation
    return np.array([x, y, pose])

def arm_random_joint_angles():
    """Generate random joint angles for the arm."""
    theta1 = np.random.uniform(0, 2 * np.pi)
    theta2 = np.random.uniform(0, 2 * np.pi)
    return np.array([theta1, theta2])

def arm_random_base_position():
    """Generate a random base position for the arm."""
    x = np.random.uniform(0, 20)
    y = np.random.uniform(0, 20)
    return np.array([x, y])

def main():
    args = parse_arguments()
    env = environment_to_array(scene_from_file(args.map))
    if env.size == 0:
        print("Environment data is empty. Exiting.")
        return

    robot_type = args.robot
    
    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_aspect('equal')

    # Plot the obstacles once at the beginning
    for idx, obstacle in enumerate(env):
        x, y, w, h, pose = obstacle
        edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                          [w / 2, h / 2], [-w / 2, h / 2]])
        transformed_edges = h_transform(edges, x, y, pose)
        polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='green', alpha=0.5)
        ax.add_patch(polygon)

    for iteration in range(1, 11):
        if robot_type == "freeBody":
            # Generate random pose for the free body
            freebody_pose = freebody_random_pose()
            x, y, pose = freebody_pose
            # Define a free body as a rectangle with fixed width and height
            w, h = 0.5, 0.3
            edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                              [w / 2, h / 2], [-w / 2, h / 2]])
            freebody_vertices = h_transform(edges, x, y, pose)

            robot_in_collision = False
            colliding_indices = []
            for idx, obstacle in enumerate(env):
                obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
                obs_edges = np.array([[-obs_w / 2, -obs_h / 2],
                                      [obs_w / 2, -obs_h / 2],
                                      [obs_w / 2, obs_h / 2],
                                      [-obs_w / 2, obs_h / 2]])
                obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)

                # Check collision of the free body with each obstacle
                if polygons_collide(freebody_vertices, obs_vertices):
                    colliding_indices.append(idx)
                    robot_in_collision = True

            # Freebody color based on collision status
            freebody_color = 'red' if robot_in_collision else 'blue'

            # Plot the free body
            polygon = plt.Polygon(freebody_vertices, edgecolor='black', facecolor=freebody_color, alpha=0.7)
            ax.add_patch(polygon)

            # If a collision occurred, turn the colliding obstacles red
            if robot_in_collision:
                for idx in colliding_indices:
                    x, y, w, h, pose = env[idx]
                    edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                                      [w / 2, h / 2], [-w / 2, h / 2]])
                    transformed_edges = h_transform(edges, x, y, pose)
                    polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='red', alpha=0.5)
                    ax.add_patch(polygon)
        elif robot_type == "arm":
            # Generate random joint angles and base position for the arm
            joint_angles = arm_random_joint_angles()
            base_position = np.array([10, 10]) #Have Arm in the middle of environment
            #base_position = arm_random_base_position()
            arm_positions = arm_forward_kinematics(joint_angles, base_position)

            robot_in_collision = False
            colliding_indices = []

            # Plot the arm
            for i in range(len(arm_positions) - 1):
                x1, y1 = arm_positions[i]
                x2, y2 = arm_positions[i + 1]
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=3)

            # Check for collisions with obstacles
            for idx, obstacle in enumerate(env):
                obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
                obs_edges = np.array([[-obs_w / 2, -obs_h / 2],
                                      [obs_w / 2, -obs_h / 2],
                                      [obs_w / 2, obs_h / 2],
                                      [-obs_w / 2, obs_h / 2]])
                obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)

                # Check collision of each link with each obstacle
                for i in range(len(arm_positions) - 1):
                    link_vertices = np.array([arm_positions[i], arm_positions[i + 1]])
                    if polygons_collide(link_vertices, obs_vertices):
                        colliding_indices.append(idx)
                        robot_in_collision = True

            # If a collision occurred, turn the colliding obstacles red
            if robot_in_collision:
                for idx in colliding_indices:
                    x, y, w, h, pose = env[idx]
                    edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                                      [w / 2, h / 2], [-w / 2, h / 2]])
                    transformed_edges = h_transform(edges, x, y, pose)
                    polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='red', alpha=0.5)
                    ax.add_patch(polygon)


        # Pause to simulate time delay without clearing previous free bodies
        plt.pause(1)

    plt.title("Free Body Positions Over Time")
    plt.grid(True)
    plt.savefig("media/collision_checking.png")
    plt.show()

if __name__ == "__main__":
    main()
