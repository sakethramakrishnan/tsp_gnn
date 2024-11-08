import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
import argparse
import pprint as pp
import time

if __name__ == "__main__":
    # Parse arguments for the number of samples and nodes
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_nodes", type=int, default=15)
    parser.add_argument("--filename", type=str, default="tsp_dynamic_programming_output.txt")
    opts = parser.parse_args()
    
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Open file for saving the results
    with open(opts.filename, "w") as f:
        start_time = time.time()
        
        for sample in range(opts.num_samples):
            # Generate random coordinates
            coordinates = np.random.random((opts.num_nodes, 2))
            
            # Calculate distance matrix
            distance_matrix = np.linalg.norm(
                coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1
            )
            
            # Solve the TSP problem
            permutation, total_distance = solve_tsp_dynamic_programming(distance_matrix)
            
            # Save nodes, solution order, and total distance
            f.write(f"Sample {sample + 1}:\n")
            f.write("Coordinates:\n")
            for coord in coordinates:
                f.write(f"{coord[0]:.4f}, {coord[1]:.4f}\n")
            f.write("Distance Matrix:\n")
            f.write(np.array2string(distance_matrix, formatter={'float_kind':lambda x: "%.4f" % x}))
            f.write("\nOptimal Path Order: " + " -> ".join(map(str, permutation)) + "\n")
            f.write(f"Total Distance: {total_distance:.4f}\n\n")
        
        end_time = time.time() - start_time
    
    print(f"Completed generation of {opts.num_samples} TSP samples.")
    print(f"Total time: {end_time:.2f}s")
