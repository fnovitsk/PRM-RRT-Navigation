import time
import numpy as np
import pandas as pd
from prm import build_prm, find_path, visualizeSamples, visualizeCSpace
from rrt import run_rrt, animate_rrt, visualize_cspace, visualize_samples
from generating_environment import scene_from_file
from collision_checking import environment_to_array

# Constants
NUM_RUNS = 10
MAX_ITERATIONS = 500
NUM_ENVIRONMENTS = 5
NUM_START_GOAL_PAIRS = 5
GOAL_RADIUS = 0.5  # Radius to consider goal reached

# Function to perform experimental evaluation of planners
def evaluate_planners():
    # Data collection lists
    data = []

    # Loop through environments
    for env_idx in range(NUM_ENVIRONMENTS):
        # Load environment
        env = scene_from_file(f"environments/environment{env_idx+1}.txt")
        env_array = environment_to_array(env)

        # Loop through different start and goal pairs
        for pair_idx in range(NUM_START_GOAL_PAIRS):
            start = np.random.uniform(0, 20, size=3)  # Example start configuration (x, y, theta)
            goal = np.random.uniform(0, 20, size=3)   # Example goal configuration (x, y, theta)

            # Evaluate PRM and RRT for each pair
            for planner in ["PRM", "RRT"]:
                success_count = 0
                total_path_quality = 0
                total_computation_time = 0

                for run_idx in range(NUM_RUNS):
                    start_time = time.time()

                    if planner == "PRM":
                        # Build PRM roadmap
                        nodes = build_prm("freeBody", env, start, goal)
                        start_node = nodes[0]
                        goal_node = nodes[1]
                        path = find_path(start_node, goal_node, nodes)
                    elif planner == "RRT":
                        # Execute RRT
                        nodes, parent, path = run_rrt(start, goal, env_array, GOAL_RADIUS, max_nodes=MAX_ITERATIONS, robot_type="freeBody")

                    # Record computation time
                    computation_time = time.time() - start_time
                    total_computation_time += computation_time

                    # Check success and calculate path quality
                    if path is not None and len(path) > 1:
                        success_count += 1
                        if planner == "PRM":
                            path_length = sum(
                                np.linalg.norm(np.array(path[i].config) - np.array(path[i + 1].config))
                                for i in range(len(path) - 1)
                            )
                        elif planner == "RRT":
                            path_length = sum(
                                np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
                                for i in range(len(path) - 1)
                            )
                        total_path_quality += path_length

                # Calculate averages
                average_success_rate = success_count / NUM_RUNS
                average_path_quality = total_path_quality / success_count if success_count > 0 else None
                average_computation_time = total_computation_time / NUM_RUNS

                # Store data
                data.append({
                    "Environment": env_idx,
                    "Start/Goal Pair": pair_idx,
                    "Planner": planner,
                    "Success Rate": average_success_rate,
                    "Average Path Quality": average_path_quality,
                    "Average Computation Time": average_computation_time
                })

    # Convert to DataFrame and save results
    df = pd.DataFrame(data)
    df_prm = df[df["Planner"]=="PRM"]
    df_rrt = df[df["Planner"]=="RRT"]
    
    df_prm = df_prm[["Success Rate", "Average Path Quality", "Average Computation Time"]].mean()
    df_rrt = df_rrt[["Success Rate", "Average Path Quality", "Average Computation Time"]].mean()

    print("PRM: ", "Avg. Success Rate: ", df_prm["Success Rate"].mean(), "Avg. Path Quality: ", 
    df_prm["Average Path Quality"].mean(), "Average Computation Time: ", df_prm["Average Computation Time"].mean())
    print("RRT: ", "Avg. Success Rate: ", df_rrt["Success Rate"].mean(), "Avg. Path Quality: ", 
    df_rrt["Average Path Quality"].mean(), "Average Computation Time: ", df_rrt["Average Computation Time"].mean())

# Main function to run evaluation
def main():
    evaluate_planners()

if __name__ == "__main__":
    main()
