import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt
#import pdb;
M = 50 #NUmber of random configurations

def generate_configs(filename: str, num_configs: int, config_length: int):
    with open(filename, 'w') as file:
        for _ in range(num_configs):
            if config_length == 2 : #Arm
                config = [round(random.uniform(0, 2 * math.pi), 2) for _ in range(2)]
            elif config_length == 3 : #Freebody
                x = round(random.uniform(0, 20), 2)
                y = round(random.uniform(0, 20), 2)
                rotation = round(random.uniform(0, 2 * math.pi), 2)
                config = [x, y, rotation]
            file.write(" ".join(map(str, config)) + "\n")


# Example usage to generate a configs.txt file for both 'arm' and 'freeBody' configurations
# For 'arm', we have 2 values per configuration, and for 'freeBody', we have 3 values per configuration

# Generate configs for 'arm'
generate_configs('configs_arm.txt', num_configs=M, config_length=2)

# Generate configs for 'freeBody'
generate_configs('configs_freeBody.txt', num_configs=M, config_length=3)

#print("Configuration files generated: configs_arm.txt and configs_freeBody.txt")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Nearest neighbors with linear search approach.")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Defines the robot to use: 'arm' or 'freeBody'.")
    parser.add_argument("--target", type=float, nargs='+', required=True, help="Target configuration for the robot. N numbers defining the configuration.")
    parser.add_argument("-k", type=int, required=True, help="Number of nearest neighbors to output.")
    parser.add_argument("--configs", type=str, required=True, help="Filename that contains random configurations.")
    return parser.parse_args()

def load_configurations(filename):
    configurations = []
    with open(filename, 'r') as file:
        for line in file:
            config = list(map(float, line.strip().split()))
            configurations.append(config)
    return np.array(configurations)

def calculate_distance(config1, config2):
    return np.linalg.norm(np.array(config1) - np.array(config2))

def find_nearest_neighbors(target, configurations, k):
    distances = []
    for i, config in enumerate(configurations):
        dist = calculate_distance(target, config)
        distances.append((dist, i))
    #pdb.set_trace()
    distances.sort(key=lambda x: x[0])
    return [configurations[idx] for _, idx in distances[:k]]

def visualize_environment_arm(target, configurations):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(0, 2 * math.pi)
    ax.set_aspect('equal')
    ax.set_title('C-Space (Arm Robot)')
    ax.set_xlabel('Theta1 (First Joint Angle in Rad)')
    ax.set_ylabel('Theta2 (Second Joint Angle in Rad)')

    # Plot target configuration in green
    if len(target) == 3:  # Assuming target is x, y, and orientation for 'freeBody'
        ax.plot(target[0], target[1], 'go', label='Target (Green)')
    elif len(target) == 2:  # Assuming target represents angles for 'arm'
        ax.plot(target[0], target[1], 'go', label='Target (Green)')

    # Plot all configurations in red
    for config in configurations:
        if len(config) == 3:  # Assuming configuration is x, y, and orientation for 'freeBody'
            ax.plot(config[0], config[1], 'ro', label='Configuration (Red)')
        elif len(config) == 2:  # Assuming configuration represents angles for 'arm'
            ax.plot(config[0], config[1], 'ro', label='Configuration (Red)')

    #plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("media/nearest_neighborsVisualization_arm.png")

def visualize_environment_freebody(target, configurations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 2 * math.pi)
    ax.set_title('C-Space (FreeBody Robot)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Rotation (in Rad)')

    # Plot target configuration in green
    ax.scatter(target[0], target[1], target[2], color='g', label='Target (Green)')

    # Plot all configurations in red
    for config in configurations:
        ax.scatter(config[0], config[1], config[2], color='r', label='Configuration (Red)')

    #plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("media/nearest_neighborsVisualization_freebody.png")

def main():
    #pdb.set_trace()
    args = parse_arguments()
    #pdb.set_trace()
    # Load configurations from file
    configurations = load_configurations(args.configs)

    # Ensure target configuration length matches expected for the robot type
    target_length = 2 if args.robot == 'arm' else 3
    if len(args.target) != target_length:
        raise ValueError(f"Target configuration must have {target_length} values for robot type '{args.robot}'.")

    # Find the k nearest neighbors
    #pdb.set_trace()
    nearest_neighbors = find_nearest_neighbors(args.target, configurations, args.k)

    print("Target Configuration:", args.target)

    # Output the nearest neighbors
    print("Nearest Neighbors:")
    for neighbor in nearest_neighbors:
        print(" ".join(map(str, neighbor)))

    if args.robot == 'arm':
        visualize_environment_arm(args.target, nearest_neighbors)
    elif args.robot =='freeBody':
        visualize_environment_freebody(args.target, nearest_neighbors)
    
if __name__ == "__main__":
    main()





