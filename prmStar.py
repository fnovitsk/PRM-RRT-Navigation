import argparse
import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from generating_environment import scene_from_file, Environment
from nearest_neighbors import load_configurations, find_nearest_neighbors
from collision_checking import environment_to_array, polygons_collide, arm_forward_kinematics, freebody_random_pose, arm_random_joint_angles, h_transform
from prm import visualizeCSpace, visualizeSamples, createAnimation
import matplotlib.animation as animation

# Constants
MAX_NODES = 1000
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
                                  [obs_w / 2, obs_h / 2], [-obs_w / 2, -obs_h / 2]])
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
                                  [obs_w / 2, obs_h / 2], [-obs_w / 2, -obs_h / 2]])
            obs_vertices = h_transform(obs_edges, obs_x, obs_y, obs_pose)
            if polygons_collide(transformed_edges, obs_vertices):
                return False
    return True

# Function to build the roadmap
def build_prm_star(robot_type, env, start, goal, radius):
    G = nx.Graph()
    nodes = [Node(start), Node(goal)]
    G.add_node(nodes[0])
    G.add_node(nodes[1])
    while len(nodes) < MAX_NODES:
        new_configs = generate_random_config(robot_type)
        for new_config in new_configs:
            if len(nodes) >= MAX_NODES:
                break
            if is_valid_node(new_config, env, robot_type):
                new_node = Node(new_config)
                nodes.append(new_node)
                G.add_node(new_node)
                # Connect to nearby nodes within radius
                nearby_nodes = find_nodes_within_radius(new_node, nodes, radius)
                for nearby_node in nearby_nodes:
                    if is_valid_edge(new_node, nearby_node, env, robot_type):
                        G.add_edge(new_node, nearby_node)
    return nodes

# Function to find neighbors within a given radius
def find_nodes_within_radius(node, nodes, radius):
    neighbors = []
    for other_node in nodes:
        if other_node != node:
            dist = np.linalg.norm(np.array(node.config) - np.array(other_node.config))
            if dist <= radius:
                neighbors.append(other_node)
    return neighbors

# Collision checking for edges
def is_valid_edge(node1, node2, env, robot_type):
    # Linear interpolation to check for collision between node1 and node2
    steps = 10  # Number of steps for interpolation
    for i in range(steps + 1):
        t = i / steps
        interpolated_config = (1 - t) * np.array(node1.config) + t * np.array(node2.config)
        if not is_valid_node(interpolated_config, env, robot_type):
            return False
    return True

# Function to find the path from start to goal using the PRM*
def find_path(start_node, goal_node, nodes, radius):
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

        neighbors = find_nodes_within_radius(current_node, nodes, radius)
        for neighbor in neighbors:
            if neighbor not in came_from:
                came_from[neighbor] = current_node
                open_set.append(neighbor)

    return None

# Main function
def main():
    parser = argparse.ArgumentParser(description="PRM* Algorithm Implementation.")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True,
                        help="Defines the robot to use: 'arm' or 'freeBody'.")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration for the robot.")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration for the robot.")
    parser.add_argument("--map", type=str, required=True, help="Filename of the map containing obstacles.")
    parser.add_argument("--goal_rad", type=float, required=True, help="Radius for connecting nodes in PRM*.")
    args = parser.parse_args()

    # Load environment and validate inputs
    env = scene_from_file(args.map)
    robot_type = args.robot
    start = args.start
    goal = args.goal
    radius = args.goal_rad

    if len(start) != len(goal):
        raise ValueError("Start and goal configurations must have the same length.")

    # Build PRM* roadmap
    nodes = build_prm_star(robot_type, env, start, goal, radius)

    # Find the path from start to goal
    start_node = nodes[0]
    goal_node = nodes[1]
    path = find_path(start_node, goal_node, nodes, radius)

    # Visualize Samples in Environment
    visualizeSamples(nodes, robot_type, env, "media/prmStarSamples.png")

    # Visualize C-Space
    visualizeCSpace(nodes, start, goal, robot_type, path, "media/prmStarCSpace.png")

    # Create animation of the robot moving along the path
    if path is not None:
        createAnimation(path, robot_type, env, "media/prmStarAnimation.gif")
    else:
        print("Path not found. No new animation created.")
    

if __name__ == "__main__":
    main()
