import argparse
import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from generating_environment import scene_from_file, Environment
from nearest_neighbors import load_configurations, find_nearest_neighbors
from collision_checking import environment_to_array, polygons_collide, arm_forward_kinematics, freebody_random_pose, arm_random_joint_angles, h_transform
import matplotlib.animation as animation

# Constants
MAX_NODES = 500
K_NEIGHBORS = 6

# PRM Node class
class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

# Function to generate random configurations
def generate_random_config(robot_type, num_configs=1):
    configs = []
    for _ in range(num_configs):
        if robot_type == "arm":
            configs.append(arm_random_joint_angles())
        elif robot_type == "freeBody":
            configs.append(freebody_random_pose())
    return configs

# Collision checking function for nodes
def is_valid_node(config, env, robot_type):
    env_array = environment_to_array(env)
    if robot_type == "arm":
        base_position = np.array([10, 10])  # Assume the base position is at the center
        positions = arm_forward_kinematics(config, base_position)
        for idx, obstacle in enumerate(env_array):
            obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
            obs_edges = np.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2],
                                  [obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])
            obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)
            for i in range(len(positions) - 1):
                link_vertices = np.array([positions[i], positions[i + 1]])
                if polygons_collide(link_vertices, obs_vertices):
                    return False
    elif robot_type == "freeBody":
        x, y, pose = config
        width, height = 0.5, 0.3
        edges = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])
        transformed_edges = h_transform(edges, x, y, pose)
        for idx, obstacle in enumerate(env_array):
            obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
            obs_edges = np.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2],
                                  [obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])
            obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)
            if polygons_collide(transformed_edges, obs_vertices):
                return False
    return True

# Function to build the roadmap
def build_prm(robot_type, env, start, goal):
    nodes = [Node(start), Node(goal)]
    while len(nodes) < MAX_NODES:
        new_configs = generate_random_config(robot_type)
        for new_config in new_configs:
            if len(nodes) >= MAX_NODES:
                break
            if is_valid_node(new_config, env, robot_type):
                new_node = Node(new_config)
                nodes.append(new_node)
    return nodes

# Function to find k-nearest neighbors
def find_k_nearest_neighbors(node, nodes, k):
    distances = []
    for other_node in nodes:
        if other_node != node:
            dist = np.linalg.norm(np.array(node.config) - np.array(other_node.config))
            distances.append((dist, other_node))
    distances.sort(key=lambda x: x[0])
    return [neighbor for _, neighbor in distances[:k]]

# Function to find the path from start to goal using the PRM
def find_path(start_node, goal_node, nodes):
    open_set = [start_node]
    came_from = {}
    came_from[start_node] = None

    while open_set:
        current_node = open_set.pop(0)
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]

        neighbors = find_k_nearest_neighbors(current_node, nodes, K_NEIGHBORS)
        for neighbor in neighbors:
            if neighbor not in came_from:
                came_from[neighbor] = current_node
                open_set.append(neighbor)

    return None

# Visualization of samples in the environment
def visualizeSamples(nodes, robot_type, env, filename):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title('PRM Physical Environment')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Plot obstacles
    env_array = environment_to_array(env)
    for obstacle in env_array:
        obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
        obs_edges = np.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2],
                              [obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])
        obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)
        polygon = plt.Polygon(obs_vertices, edgecolor='red', facecolor='none')
        ax.add_patch(polygon)

    # Plot nodes (configurations)
    for i, node in enumerate(nodes):
        config = node.config
        if robot_type == 'freeBody':
            color = 'green' if i == 0 else ('red' if i == 1 else 'blue')
            alpha = 1.0 if i in [0, 1] else 0.3
            width, height = 0.5, 0.3
            edges = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                              [width / 2, height / 2], [-width / 2, height / 2]])
            transformed_edges = h_transform(edges, config[0], config[1], config[2])
            polygon = plt.Polygon(transformed_edges, edgecolor=color, facecolor='none', alpha=alpha, label=('Start' if i == 0 else 'Goal' if i == 1 else None))
            ax.add_patch(polygon)
        elif robot_type == 'arm':
            base_position = np.array([10, 10])
            arm_positions = arm_forward_kinematics(config, base_position)
            color = 'green' if i == 0 else ('red' if i == 1 else 'blue')
            alpha = 1.0 if i in [0, 1] else 0.3
            for j in range(len(arm_positions) - 1):
                x1, y1 = arm_positions[j]
                x2, y2 = arm_positions[j + 1]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=2 if i in [0, 1] else 1)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Visualization of the C-space
def visualizeCSpace(nodes, start, goal, robot_type, path, filename):
    if robot_type == 'arm':
        fig, ax = plt.subplots()
        ax.set_xlim(0, 2 * math.pi)
        ax.set_ylim(0, 2 * math.pi)
        ax.set_aspect('equal')
        ax.set_title('C-Space (Arm Robot)')
        ax.set_xlabel('Theta1 (First Joint Angle in Rad)')
        ax.set_ylabel('Theta2 (Second Joint Angle in Rad)')

        ax.plot(start[0], start[1], 'go', label='Start')
        ax.plot(goal[0], goal[1], 'ro', label='Goal')
        
        for node in nodes[2:]:
            config = node.config
            ax.plot(config[0], config[1], 'bo', alpha=0.3)

        for node in nodes:
            neighbors = find_k_nearest_neighbors(node, nodes, K_NEIGHBORS)
            for neighbor in neighbors:
                ax.plot([node.config[0], neighbor.config[0]], [node.config[1], neighbor.config[1]], 'k-', linewidth=0.5, alpha=0.1)

        if path is not None:
            for i in range(len(path) - 1):
                start_config = path[i].config
                end_config = path[i + 1].config
                ax.plot([start_config[0], end_config[0]], [start_config[1], end_config[1]], 'g-', linewidth=2, alpha=0.7)

        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(filename)
        plt.show()

    elif robot_type == 'freeBody':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 2 * math.pi)
        ax.set_title('C-Space (FreeBody Robot)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Rotation (in Rad)')

        ax.scatter(start[0], start[1], start[2], color='g', label='Start (Green)')
        ax.scatter(goal[0], goal[1], goal[2], color='r', label='Goal (Red)')

        for node in nodes[2:]:
            config = node.config
            ax.scatter(config[0], config[1], config[2], color='b', alpha=0.3)

        for node in nodes:
            neighbors = find_k_nearest_neighbors(node, nodes, K_NEIGHBORS)
            for neighbor in neighbors:
                ax.plot([node.config[0], neighbor.config[0]], [node.config[1], neighbor.config[1]], [node.config[2], neighbor.config[2]], 'k-', linewidth=0.5, alpha=0.1)

        if path is not None:
            for i in range(len(path) - 1):
                start_config = path[i].config
                end_config = path[i + 1].config
                ax.plot([start_config[0], end_config[0]], [start_config[1], end_config[1]], [start_config[2], end_config[2]], 'g-', linewidth=2, alpha=0.7)

        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.show()

# Create animation of the robot moving through the path
def createAnimation(path, robot_type, env, filename):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title('Robot Path Animation')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Plot obstacles once
    env_array = environment_to_array(env)
    for obstacle in env_array:
        obs_x, obs_y, obs_w, obs_h, obs_pose = obstacle
        obs_edges = np.array([[-obs_w / 2, -obs_h / 2], [obs_w / 2, -obs_h / 2],
                              [obs_w / 2, obs_h / 2], [-obs_w / 2, obs_h / 2]])
        obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)
        polygon = plt.Polygon(obs_vertices, edgecolor='red', facecolor='none')
        ax.add_patch(polygon)


    # Function to update the plot at each frame
    def update(frame):
        # Remove previous robot plot (if any)
        for artist in ax.findobj(match=plt.Polygon):
            if artist.get_edgecolor() == 'blue':
                artist.remove()
        for artist in ax.findobj(match=plt.Line2D):
            if artist.get_color() == 'blue':
                artist.remove()

        # Plot robot at current frame position
        config = path[frame].config
        if robot_type == 'freeBody':
            width, height = 0.5, 0.3
            edges = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                              [width / 2, height / 2], [-width / 2, height / 2]])
            transformed_edges = h_transform(edges, config[0], config[1], config[2])
            polygon = plt.Polygon(transformed_edges, edgecolor='blue', facecolor='none')
            ax.add_patch(polygon)
        elif robot_type == 'arm':
            base_position = np.array([10, 10])
            arm_positions = arm_forward_kinematics(config, base_position)
            for j in range(len(arm_positions) - 1):
                x1, y1 = arm_positions[j]
                x2, y2 = arm_positions[j + 1]
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2)

    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(filename, writer='imagemagick')
    plt.show()


# Main function
def main():
    parser = argparse.ArgumentParser(description="PRM Algorithm Implementation.")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True,
                        help="Defines the robot to use: 'arm' or 'freeBody'.")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration for the robot.")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration for the robot.")
    parser.add_argument("--map", type=str, required=True, help="Filename of the map containing obstacles.")
    args = parser.parse_args()

    env = scene_from_file(args.map)
    robot_type = args.robot
    start = args.start
    goal = args.goal

    if len(start) != len(goal):
        raise ValueError("Start and goal configurations must have the same length.")

    nodes = build_prm(robot_type, env, start, goal)

    start_node = nodes[0]
    goal_node = nodes[1]
    path = find_path(start_node, goal_node, nodes)
    if robot_type == 'freeBody':
        visualizeSamples(nodes, robot_type, env, "media/prmFreeBodySamples.png")
        visualizeCSpace(nodes, start, goal, robot_type, path, "media/prmFreeBodyCSpace.png")
        if path is not None:
            createAnimation(path, robot_type, env, "media/prmFreeBodyAnimation.gif")
    
    if robot_type == 'arm':
        visualizeSamples(nodes, robot_type, env, "media/prmArmSamples.png")
        visualizeCSpace(nodes, start, goal, robot_type, path, "media/prmArmCSpace.png")
        if path is not None:
            createAnimation(path, robot_type, env, "media/prmArmAnimation.gif")


    
if __name__ == "__main__":
    main()
