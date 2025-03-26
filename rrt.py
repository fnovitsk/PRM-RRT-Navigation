import argparse
import random
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from generating_environment import scene_from_file, Environment
from scipy.spatial import KDTree
from collision_checking import environment_to_array, h_transform, polygons_collide, arm_forward_kinematics

def parse_arguments():
    parser = argparse.ArgumentParser(description="RRT Path Planning.")
    parser.add_argument("--robot", type=str, choices=["arm","freeBody"], required=True, help="Defines the robot to use: 'arm' or 'freeBody'.")
    parser.add_argument("--map", type=str, required=True, help="Filename of the map containing obstacles.")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration of the robot.")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration of the robot.")
    parser.add_argument("--goal_rad", type=float, required=True, help="Radius to consider goal reached.")
    return parser.parse_args()

def is_collision_free(point, env, robot_type="freeBody"):
    if robot_type == "freeBody":
        # For freeBody, consider the robot as a 0.5x0.3 rectangle
        x, y, theta = point
        width, height = 0.5, 0.3
        edges = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
        robot_vertices = h_transform(edges, x, y, theta)
    elif robot_type == "arm":
        # For arm robot, get joint positions
        joint_angles = point
        base_position = np.array([10, 10])  # Assuming base is at the center of the environment
        robot_vertices = arm_forward_kinematics(joint_angles, base_position)

    for obstacle in env:
        x, y, w, h, pose = obstacle
        edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        obstacle_vertices = h_transform(edges, x, y, pose)
        if polygons_collide(robot_vertices, obstacle_vertices):
            return False
    return True

def edge_collision_free(point1, point2, env, robot_type="freeBody", num_checks=10):
    interpolated_points = np.linspace(point1, point2, num_checks)
    for point in interpolated_points:
        if not is_collision_free(point, env, robot_type):
            return False
    return True

def run_rrt(start, goal, env, goal_rad, max_nodes=1000, seed=42, robot_type="freeBody", link_lengths=[2.0, 2.0]):
    np.random.seed(seed)
    random.seed(seed)
    nodes = [start]
    parent = {tuple(start): None}
    kdtree = KDTree(np.array(nodes))

    for i in range(max_nodes):
        goal_bias = 0.05 + (0.5 - 0.05) * (i / max_nodes)  # Gradually increase goal bias from 0.05 to 0.5
        if random.random() < goal_bias:
            rand_point = goal
        else:
            if robot_type == "arm":
                rand_point = np.random.uniform(0, 2 * np.pi, size=len(start))
                # Avoid clustering by ensuring unique samples and adding perturbations
                while tuple(rand_point) in parent:
                    rand_point = np.random.uniform(0, 2 * np.pi, size=len(start))
                rand_point += np.random.uniform(-0.1, 0.1, size=len(start))  # Add a small perturbation
            else:
                rand_point = np.array([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(0, 2 * np.pi)])

        if len(nodes) > 1:
            kdtree = KDTree(np.array(nodes))

        _, nearest_idx = kdtree.query(rand_point)
        nearest_point = nodes[nearest_idx]

        if robot_type == "arm":
            # For arm robot, apply adaptive incremental joint rotations
            delta_theta = 0.05 + 0.15 * (i / max_nodes)  # Adaptive step size
            direction = rand_point - nearest_point
            norm_direction = direction / np.linalg.norm(direction)
            new_joint_angles = nearest_point + norm_direction * delta_theta
            new_point = new_joint_angles

            # Prioritize new configurations that improve overall reach and minimize redundant sampling
            if np.linalg.norm(new_point - goal) < np.linalg.norm(nearest_point - goal):
                new_point += np.random.uniform(-0.05, 0.05, size=len(start))  # Add a small perturbation to improve distribution
        else:
            # For freeBody robot, move towards the random point with (x, y, theta)
            direction = (rand_point - nearest_point) / np.linalg.norm(rand_point - nearest_point)
            step_size = min(1.5, np.linalg.norm(goal[:2] - nearest_point[:2]) / 1.5)  # Adjusted step size for controlled growth
            new_position = nearest_point[:2] + direction[:2] * step_size
            new_theta = nearest_point[2] + (rand_point[2] - nearest_point[2]) * 0.1 if len(nearest_point) == 3 else 0.0  # Adjust theta incrementally
            new_point = np.array([new_position[0], new_position[1], new_theta]) if len(nearest_point) == 3 else np.array([new_position[0], new_position[1]])

        # Check if the edge between nearest_point and new_point is collision-free
        if edge_collision_free(nearest_point, new_point, env, robot_type=robot_type):
            nodes.append(new_point)
            parent[tuple(new_point)] = tuple(nearest_point)

            if robot_type == "arm":
                base_position = np.array([10, 10])
                if np.linalg.norm(new_point - goal) < goal_rad:
                    # Path smoothing step
                    path = [goal]
                    current = tuple(new_point)
                    while current is not None:
                        path.append(current)
                        current = parent.get(current)
                    path = path[::-1]

                    # Smoothing the path by attempting to shortcut nodes
                    smoothed_path = [path[0]]
                    for i in range(1, len(path) - 1):
                        if not edge_collision_free(smoothed_path[-1], path[i + 1], env, robot_type=robot_type):
                            smoothed_path.append(path[i])
                    smoothed_path.append(path[-1])

                    return nodes, parent, smoothed_path
            else:
                if np.linalg.norm(new_point[:2] - goal[:2]) < goal_rad and abs(new_point[2] - goal[2]) < 0.1:
                    # Path smoothing step
                    path = [goal]
                    current = tuple(new_point)
                    while current is not None:
                        path.append(current)
                        current = parent.get(current)
                    path = path[::-1]

                    # Smoothing the path by attempting to shortcut nodes
                    smoothed_path = [path[0]]
                    for i in range(1, len(path) - 1):
                        if not edge_collision_free(smoothed_path[-1], path[i + 1], env, robot_type=robot_type):
                            smoothed_path.append(path[i])
                    smoothed_path.append(path[-1])

                    return nodes, parent, smoothed_path

    # If goal not reached, still return nodes, parent, and an empty path
    return nodes, parent, []

def animate_rrt(nodes, parent, env, start, goal, path=None, robot_type="freeBody", link_lengths=[2.0, 2.0]):
    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_aspect('equal')

    # Plot environment
    for obstacle in env:
        x, y, w, h, pose = obstacle
        edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        transformed_edges = h_transform(edges, x, y, pose)
        polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='green', alpha=0.5)
        ax.add_patch(polygon)

    tree_lines = []
    if robot_type == 'freeBody':
        robot_rect = plt.Rectangle((start[0] - 0.25, start[1] - 0.15), 0.5, 0.3, angle=start[2] * 180 / np.pi, color='black', label='Robot')
    elif robot_type == 'arm':
        # Represent the arm robot as a series of links
        arm_positions = arm_forward_kinematics(start, np.array([10, 10]))  # Set start positions as base position (10,10)
        robot_lines, = ax.plot(arm_positions[:, 0], arm_positions[:, 1], color='black', label='Robot')
    if robot_type == 'freeBody':
        ax.add_patch(robot_rect)
    if robot_type == 'freeBody':
        start_marker = plt.Rectangle((start[0] - 0.25, start[1] - 0.15), 0.5, 0.3, angle=start[2] * 180 / np.pi, color='red', label='Start', zorder=5)
    else:
        arm_start_positions = arm_forward_kinematics(start, np.array([10, 10]))
        start_marker, = ax.plot(arm_start_positions[:, 0], arm_start_positions[:, 1], 'r-', label='Start', zorder=5)  # Updated start visualization
    if robot_type == 'freeBody':
        ax.add_patch(start_marker)
    else:
        ax.add_artist(start_marker)

    if robot_type == 'freeBody':
        goal_marker = plt.Rectangle((goal[0] - 0.25, goal[1] - 0.15), 0.5, 0.3, angle=goal[2] * 180 / np.pi, color='green', linestyle='dashed', label='Goal', zorder=5)
    else:
        arm_goal_positions = arm_forward_kinematics(goal, np.array([10, 10]))
        goal_marker, = ax.plot(arm_goal_positions[:, 0], arm_goal_positions[:, 1], color='green', linestyle='dashed', label='Goal', zorder=5)
    if robot_type == 'freeBody':
        ax.add_patch(goal_marker)
    else:
        ax.add_artist(goal_marker)

    def init():
        if robot_type == 'freeBody':
            return tree_lines + [robot_rect, start_marker, goal_marker]
        elif robot_type == 'arm':
            return tree_lines + [robot_lines, start_marker, goal_marker]

    def update(frame, robot_type=robot_type, path=path):
        if frame < len(nodes):
            # Draw RRT expansion lines
            node = nodes[frame]
            if parent[tuple(node)] is not None:
                nearest_node = parent[tuple(node)]
                line, = ax.plot([nearest_node[0], node[0]], [nearest_node[1], node[1]], color='blue', linewidth=0.5, alpha=0.5)
                tree_lines.append(line)
        elif path is not None and len(path) > 1:
            # Smoothly move along the path if a valid path exists
            idx = (frame - len(nodes)) // 20
            if idx < len(path) - 1:
                interp_fraction = (frame - len(nodes)) % 20 / 20.0
                start_pos = np.array(path[idx])
                end_pos = np.array(path[idx + 1])
                position = start_pos + interp_fraction * (end_pos - start_pos)
                if robot_type == 'arm':
                    arm_positions = arm_forward_kinematics(position, np.array([10, 10]))
                    robot_lines.set_data(arm_positions[:, 0], arm_positions[:, 1])
                elif robot_type == 'freeBody':
                    robot_rect.set_xy((position[0] - 0.25, position[1] - 0.15))
                    robot_rect.set_angle(position[2] * 180 / np.pi)

        if robot_type == 'freeBody':
            return tree_lines + [robot_rect, start_marker, goal_marker]
        elif robot_type == 'arm':
            return tree_lines + [robot_lines, start_marker, goal_marker]

    # Animate only the RRT expansion if no valid path is found
    total_frames = len(nodes) if path is None or len(path) < 2 else len(nodes) + (len(path) - 1) * 20
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, repeat=False)
    plt.legend(loc='upper left')
    plt.title("RRT Path Planning Animation")
    plt.show()

    if path is not None and len(path) > 1:
        if robot_type == "freeBody":
            ani.save("media/rrt_freeBody_animation.mp4", writer='ffmpeg', fps=30)
        if robot_type == "arm":
            ani.save("media/rrt_arm_animation.mp4", writer='ffmpeg', fps=30)
        plt.show()

def visualize_cspace(target, configurations, smoothed_path, robot_type="freeBody"):
    if robot_type == "freeBody":
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

        # Plot start configuration in red
        ax.scatter(configurations[0][0], configurations[0][1], configurations[0][2], color='r', label='Start (Red)')

        # Plot all other configurations in pink with lower opacity, but only label the first instance to avoid clutter
        ax.scatter(configurations[1][0], configurations[1][1], configurations[1][2], color='pink', alpha=0.3, label='Configuration (Pink)')
        for config in configurations[2:]:
            ax.scatter(config[0], config[1], config[2], color='pink', alpha=0.3)

        # Plot optimal path from start to goal in blue
        if len(smoothed_path) > 1:
            path_points = np.array(smoothed_path)
            ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='blue', label='Path (Blue)')
        
        plt.grid(True)
        plt.legend()
        plt.show()
        fig.savefig("media/rrt_freeSpace_cspace.png")
    elif robot_type == "arm":
        fig, ax = plt.subplots()
        ax.set_xlim(0, 2 * math.pi)
        ax.set_ylim(0, 2 * math.pi)
        ax.set_title('C-Space (Arm Robot)')
        ax.set_xlabel('Theta1 (in Rad)')
        ax.set_ylabel('Theta2 (in Rad)')

        # Plot target configuration in green
        ax.plot(target[0], target[1], 'go', label='Target (Green)')

        # Plot start configuration in red
        ax.plot(configurations[0][0], configurations[0][1], 'ro', label='Start (Red)')

        # Plot all other configurations in pink with lower opacity, but only label the first instance to avoid clutter
        ax.plot(configurations[1][0], configurations[1][1], 'o', color='pink', alpha=0.3, label='Configuration (Pink)')
        for config in configurations[2:]:
            ax.plot(config[0], config[1], 'o', color='pink', alpha=0.3)

        # Plot optimal path from start to goal in blue
        if len(smoothed_path) > 1:
            path_points = np.array(smoothed_path)
            ax.plot(path_points[:, 0], path_points[:, 1], color='blue', label='Path (Blue)')

        plt.grid(True)
        plt.legend()
        plt.show()
        fig.savefig("media/rrt_arm_cspace.png")

def visualize_samples(env, nodes, robot_type="freeBody"):
    if robot_type == 'freeBody':
        fig, ax = plt.subplots()
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        ax.set_aspect('equal')
        ax.set_title('Sample Space for FreeBody Robot')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Plot environment obstacles
        for obstacle in env:
            x, y, w, h, pose = obstacle
            edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            transformed_edges = h_transform(edges, x, y, pose)
            polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='green', alpha=0.5)
            ax.add_patch(polygon)

        # Plot RRT nodes
        nodes_array = np.array(nodes)
        ax.scatter(nodes_array[:, 0], nodes_array[:, 1], color='blue', alpha=0.3)

        # Save figure as PNG
        plt.savefig("media/rrt_freeBody_samples.png")
        plt.show()
    elif robot_type == 'arm':
        fig, ax = plt.subplots()
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        ax.set_title('Sample Space for Arm Robot')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Plot environment obstacles
        for obstacle in env:
            x, y, w, h, pose = obstacle
            edges = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            transformed_edges = h_transform(edges, x, y, pose)
            polygon = plt.Polygon(transformed_edges, edgecolor='black', facecolor='green', alpha=0.5)
            ax.add_patch(polygon)

        # Plot RRT nodes in terms of (x, y) positions for arm
        nodes_array = np.array(nodes)
        for joint_angles in nodes_array:
            arm_positions = arm_forward_kinematics(joint_angles, np.array([10, 10]))
            ax.plot(arm_positions[-1][0], arm_positions[-1][1], 'o', color='blue', alpha=0.3, label='End-Effector Position')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Save figure as PNG
        plt.savefig("media/rrt_arm_samples.png")
        plt.show()

def main():
    args = parse_arguments()
    env = environment_to_array(scene_from_file(args.map))
    start = np.array(args.start)
    goal = np.array(args.goal)
    goal_rad = args.goal_rad

    nodes, parent, smoothed_path = run_rrt(start, goal, env, goal_rad, robot_type=args.robot)
    animate_rrt(nodes, parent, env, start, goal, smoothed_path, robot_type=args.robot, link_lengths=[2.0, 2.0])
    visualize_cspace(goal, nodes, smoothed_path, robot_type=args.robot)
    visualize_samples(env, nodes, robot_type=args.robot)

if __name__ == "__main__":
    main()
