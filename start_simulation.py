"""
This module simulates federated learning for Kaplan-Meier survival analysis. It defines the 
`simulate_federated_learning`, `run_experiments`, and `plot_computation_times` functions, 
along with their respective tasks: simulating federated learning rounds, running experiments 
with varying client counts, and plotting the computational times for each client count.

"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from client import KMClient  # Import KMClient class
from server import KMServer  # Import KMServer class


def simulate_federated_learning(dataset_path, num_clients):
    """
    Simulates the federated learning process for Kaplan-Meier survival analysis.
    This function splits the dataset into client subsets, performs two rounds of
    federated learning (aggregating time points and event counts), and calculates
    the computational time taken for the entire simulation.

    Parameters:
        dataset_path (str): Path to the dataset CSV file.
        num_clients (int): The number of clients in the federated learning setup.

    Returns:
        float: The time taken for the federated learning simulation (in seconds).
    """
    start_time = time.time()  # Start the timer

    # Load the centralized dataset
    data = pd.read_csv(dataset_path, delimiter=";")

    # Split data into client datasets
    splits = uniform_split(data, num_clients)

    # Initialize clients
    clients = []
    for i in range(num_clients):
        clients.append(KMClient(i, splits[i]))

    # Initialize server
    server = KMServer(num_clients)

    # Round 1: Aggregate time points from all clients
    server.aggregate_round_1(clients)

    # Round 2: Aggregate event and at-risk counts and compute global Kaplan-Meier curve
    server.aggregate_round_2(clients)

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return elapsed_time


def uniform_split(data, num_clients):
    """
    Splits the dataset into subsets for each client. The dataset is shuffled and
    split into `num_clients` parts.

    Parameters:
        data (pd.DataFrame): The dataset to be split.
        num_clients (int): The number of clients to split the dataset into.

    Returns:
        list: A list of DataFrames, where each DataFrame corresponds to a client's dataset.

    Example:
        If there are 10 rows and 2 clients, it will split the dataset into two
        parts with 5 rows each.
    """
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return np.array_split(shuffled_data, num_clients)


def run_experiments(dataset_path, client_counts, num_runs=1):
    """
    Runs experiments for different client counts and collects the computational
    times for each client count. The simulation is repeated `num_runs` times
    to get the mean computation time.

    Parameters:
        dataset_path (str): Path to the dataset.
        client_counts (list): List of client counts to experiment with.
        num_runs (int): Number of times to run the simulation for each client count.

    Returns:
        list: List of mean computation times for each client count.
    """
    mean_times = []

    for num_clients in client_counts:
        print(f"\n=== Running experiment with {num_clients} clients ===")

        # Collect computation times for multiple runs
        times = []
        for _ in range(num_runs):
            time_taken = simulate_federated_learning(dataset_path, num_clients)
            times.append(time_taken)

        # Calculate mean for the times
        mean_time = np.mean(times)
        mean_times.append(mean_time)

        print(f"Time taken for {num_clients} clients (mean): {mean_time:.2f} seconds")

    return mean_times


def plot_computation_times(
    client_counts, mean_times, save_path="computation_times.png"
):
    """
    Plots the computational times for different numbers of clients with mean times
    and saves the graph to an image file.

    Parameters:
        client_counts (list): List of client counts.
        mean_times (list): List of mean computation times for each client count.
        save_path (str): Path to save the computation time graph.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(client_counts, mean_times, "-o", color="b", label="Computational Time")

    plt.xlabel("Number of Clients")
    plt.ylabel("Computational Time (seconds)")
    plt.title("Computational Time vs Number of Clients")
    plt.grid(True)
    plt.xticks(client_counts)

    # Save the graph as a PNG image
    plt.savefig(save_path)
    print(f"Computation time graph saved as {save_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "synthetic_data.csv"

    # Varying number of clients for the experiment
    client_counts = [2, 5, 10, 20, 30, 40, 50]  # List of client counts

    # Run the experiments and collect the computation times
    mean_times = run_experiments(dataset_path, client_counts, num_runs=1)

    # Save the computation times graph
    plot_computation_times(client_counts, mean_times, save_path="computation_times.png")
