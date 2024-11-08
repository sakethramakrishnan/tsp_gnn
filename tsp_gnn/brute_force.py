import numpy as np
import itertools
import re

def parse_file(filename):
    """
    Parses the file with structured data, extracting distance matrices for each sample.
    Returns a list of distance matrices.
    """
    distance_matrices = []
    with open(filename, "r") as f:
        content = f.read()
    
    # Split file content by each sample
    samples = content.split("Sample ")
    for sample in samples[1:]:  # Skip the first empty split

        # Extract distance matrix section between 'Distance Matrix:' and 'Optimal Path Order:'
        distance_matrix_section = re.search(r"Distance Matrix:\s*\[\[(.*?)\]\]", sample, re.S)
        if distance_matrix_section:
            matrix_data = distance_matrix_section.group(1).strip()
            
            # Read and accumulate multiline matrix data
            matrix = []
            row_buffer = []
            for line in matrix_data.splitlines():
                row_buffer.append(line.strip())
                
                # If line ends with ']', it's the end of a row
                if line.strip().endswith("]"):
                    # Join all parts of the row, remove extra characters, and parse into floats
                    full_row = ' '.join(row_buffer).replace('[', '').replace(']', '')
                    row_values = list(map(float, full_row.split()))
                    matrix.append(row_values)
                    row_buffer = []  # Clear buffer for next row

            distance_matrices.append(np.array(matrix))
    
    return distance_matrices

def brute_force_tsp(distance_matrix):
    """
    Brute-force search for the optimal TSP route using the provided distance matrix.
    """
    num_nodes = distance_matrix.shape[0]
    min_distance = float('inf')
    best_permutation = None
    
    # Generate all possible routes (permutations of nodes)
    for perm in itertools.permutations(range(num_nodes)):
        # Calculate total distance for this permutation
        current_distance = sum(
            distance_matrix[perm[i], perm[i + 1]] for i in range(num_nodes - 1)
        )
        # Add the return distance to complete the cycle
        current_distance += distance_matrix[perm[-1], perm[0]]
        
        # Update minimum distance and best permutation
        if current_distance < min_distance:
            min_distance = current_distance
            best_permutation = perm
            
    return best_permutation, min_distance

if __name__ == "__main__":
    filename = "tsp_dynamic_programming_output.txt"  # Update with your actual filename
    
    # Read and parse the file to extract distance matrices
    distance_matrices = parse_file(filename)
    
    # print(distance_matrices)
    
    print(len(distance_matrices))
    
    distance_matrices = distance_matrices[:2]
    
    # Process each distance matrix with brute-force TSP
    for i, distance_matrix in enumerate(distance_matrices):
        best_route, min_distance = brute_force_tsp(distance_matrix)
        print(f"Sample {i + 1} Brute Force Solution:")
        print(f"Best Route: {' -> '.join(map(str, best_route))}")
        print(f"Total Distance: {min_distance}\n")
