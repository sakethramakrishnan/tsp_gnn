import numpy as np
import os
import time
from python_tsp.exact import solve_tsp_dynamic_programming
import csv

# Function to generate and save coordinates
def generate_and_save_coordinates(node_counts, sample_sizes, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate coordinates for each combination of node_count and sample_size
    for n in node_counts:
        for s in sample_sizes:
            coordinates = np.random.random((n, 2))  # Generate random coordinates for `n` nodes
            file_name = f"{n}_nodes_{s}_samples_coordinates.txt"
            file_path = os.path.join(folder_path, file_name)
            
            # Save the coordinates to a file
            with open(file_path, 'w') as f:
                for coord in coordinates:
                    f.write(f"{coord[0]}, {coord[1]}\n")
            print(f"Coordinates for {n} nodes, {s} samples saved to {file_path}")

# Function to load coordinates from file
def load_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
    return np.array(coordinates)

# Function to calculate the distance matrix
def calculate_distance_matrix(coords):
    return np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

# Function to run the TSP solver and save results
def run_tsp_solver(node_counts, sample_sizes, coordinates_folder, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['x (node_count)', 'y (sample_size)', 'z (time)'])

        # Run the tests for each combination of node_count and sample_size
        for n in node_counts:
            for sample_size in sample_sizes:
                total_time = 0
                for sample in range(sample_size):
                    # Load coordinates from the corresponding file
                    coord_file = os.path.join(coordinates_folder, f"{n}_nodes_{sample_size}_samples_coordinates.txt")
                    coordinates = load_coordinates(coord_file)
                    
                    # Calculate the distance matrix
                    distance_matrix = calculate_distance_matrix(coordinates)
                    
                    # Solve the TSP problem using dynamic programming
                    start_time = time.time()
                    permutation, total_distance = solve_tsp_dynamic_programming(distance_matrix)
                    end_time = time.time()
                    
                    # Measure the time taken and accumulate it
                    total_time += (end_time - start_time)

                # Calculate the average time for this sample size
                average_time = total_time / sample_size
                writer.writerow([n, sample_size, average_time])
                print(f"  Test {n} nodes, {sample_size} samples - Average Time: {average_time:.6f} seconds")

if __name__ == "__main__":
    # Define the node counts and sample sizes
    node_counts = [2, 4, 6, 8, 10, 12, 14, 16]  # Number of nodes (cities) to test
    sample_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]  # Sample sizes to average over

    # Folder where the coordinate files will be saved and loaded from
    coordinates_folder = 'coordinates_folder'

    # Output CSV file for storing results
    output_file = 'tsp_analysis_results.csv'

    # Generate and save the coordinates (this will create 4 different files)
    generate_and_save_coordinates(node_counts, sample_sizes, coordinates_folder)
    
    # Run the TSP solver and save results to the CSV
    run_tsp_solver(node_counts, sample_sizes, coordinates_folder, output_file)
    print(f"Results have been written to {output_file}")
