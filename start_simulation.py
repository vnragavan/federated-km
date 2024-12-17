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
from he_client import KMClientHE  # Import HEKMClient class
from he_server import KMServerHE  # Import KMServer class
from centralized_km import preprocess_data
from centralized_km import plot_kaplan_meier_curve
from lifelines.statistics import logrank_test
import os

def simulate_federated_learning(dataset_path, num_clients, data_source):
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
    

    # Load the centralized dataset
    data = pd.read_csv(dataset_path, delimiter=";")

    # Split data into client datasets
    splits = uniform_split(data, num_clients)

    # Initialize clients
    clients = []
    for i in range(num_clients):
        clients.append(KMClient(i, splits[i]))

    # Initialize server
    server = KMServer(num_clients,data_source)
    
    start_time = time.time()  # Start the timer

    # Round 1: Aggregate time points from all clients
    server.aggregate_round_1(clients)

    # Round 2: Aggregate event and at-risk counts and compute global Kaplan-Meier curve
    survival_probabilities = server.aggregate_round_2(clients)

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return elapsed_time, survival_probabilities, server.global_timescale

def simulate_federated_learning_HE(dataset_path, num_clients, data_source):
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
   

    # Load the centralized dataset
    data = pd.read_csv(dataset_path, delimiter=";")

    # Split data into client datasets
    splits = uniform_split(data, num_clients)

    # max batch size of messages to be packed in encoding
    batch_size = 16

    # Initialize clients
    clients = []
    for i in range(num_clients):
        clients.append(KMClientHE(i, splits[i], batch_size))

    # Initialize server
    server = KMServerHE(num_clients, clients, batch_size, data_source)
    
    start_time = time.time()  # Start the timer

    #Key generation
    server.generate_crypto_context()
    server.generate_keys()

    # Round 1: Aggregate time points from all clients
    # no he here
    server.aggregate_round_1()

    # Round 2: Aggregate event and at-risk counts and compute global Kaplan-Meier curve
    survival_probabilities = server.aggregate_round_2_HE()

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return elapsed_time, survival_probabilities, server.global_timescale



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


def run_experiments(dataset_path, data_source, client_counts, num_runs=1):
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
    #prepare a benchmark (centralised solution)
    data = preprocess_data(dataset_path)
    central_survival_data, central_survival_prob = plot_kaplan_meier_curve(data, data_source)
    central_time = central_survival_prob.index
    #test = survival_prob.values
    #
    mean_times = []
    mean_times_he = []
    all_survival_probabilities_he = [] 
    all_global_timescale = []

    for num_clients in client_counts:
        print(f"\n=== Running experiment with {num_clients} clients ===")

        # Collect computation times for multiple runs
        times = []
        times_he = []
        for _ in range(num_runs):
            time_taken_he, survival_probabilities_he, global_timescale_he = simulate_federated_learning_HE(dataset_path, num_clients, data_source)
            time_taken, survival_probabilities, global_timescale = simulate_federated_learning(dataset_path, num_clients, data_source)
            times.append(time_taken)
            times_he.append(time_taken_he)
            all_survival_probabilities_he.append(survival_probabilities_he)

        # Calculate mean for the times
        mean_time = np.mean(times)
        mean_times.append(mean_time)
        mean_time_he = np.mean(times_he)
        mean_times_he.append(mean_time_he)

        print(f"Time taken for {num_clients} clients (mean) for {data_source}: {mean_time:.2f} seconds")
        print(f"Time taken for {num_clients} clients (mean) for {data_source} with he: {mean_time_he:.2f} seconds")

     #
    for survival_probabilities_he in all_survival_probabilities_he: 
        results = logrank_test(central_survival_prob.index, global_timescale_he, 
                       event_observed_A=central_survival_prob.values, 
                       event_observed_B=survival_probabilities_he)
        # Print the results
        print("Log-Rank Test Results:")
        print(f"Test Statistic: {results.test_statistic}")
        print(f"P-Value: {results.p_value}")
    return mean_times, mean_times_he


def plot_computation_times_comparison(
    client_counts, mean_times, mean_times_he, save_path="computation_times_comparison.png"
):
    """
    Plots the computational times as a bar graph for different numbers of clients
    comparing non-encrypted and homomorphic encryption-based computation times.

    Parameters:
        client_counts (list): List of client counts.
        mean_times (list): Mean computation times for non-encrypted Kaplan-Meier curves.
        mean_times_he (list): Mean computation times for homomorphic encryption-based Kaplan-Meier curves.
        save_path (str): Path to save the computation time comparison graph.
    """
    x = np.arange(len(client_counts))  # Position of groups
    bar_width = 0.35  # Width of the bars

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, mean_times, width=bar_width, color="b", label="Non-Encrypted KM Curves")
    plt.bar(x + bar_width / 2, mean_times_he, width=bar_width, color="g", label="HE-Based KM Curves")

    # Update Titles and Labels
    plt.xlabel("Number of Clients")
    plt.ylabel("Computational Time (seconds)")
    plt.title("Comparison of Computational Time: Non-Encrypted vs HE-Based Kaplan-Meier Curves")
    plt.xticks(x, client_counts)  # Set x-axis labels to client counts
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a legend
    plt.legend()

    # Save the graph as a PNG image
    plt.savefig(save_path)
    print(f"Computation time comparison graph saved as {save_path}")
    
    
def inspect_and_transform_dataset(file_path, censoring_col="vit_stat", survival_col="vit_stat_int"):
    """
    Inspects and transforms the dataset to ensure it follows the Kaplan-Meier convention.

    Parameters:
        file_path (str): Path to the dataset file.
        censoring_col (str): Name of the column indicating censoring or events.
        survival_col (str): Name of the survival time column.

    Returns:
        str: Path to the transformed dataset CSV file.
    """
    # Load the dataset
    dataset = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the censoring column exists
    if censoring_col not in dataset.columns:
        raise KeyError(f"Column '{censoring_col}' not found in the dataset. Available columns: {dataset.columns}")

    # Transform values in censoring column to {0, 1}
    if 2 in dataset[censoring_col].unique():
        dataset[censoring_col] = dataset[censoring_col].replace({1: 0, 2: 1})
        print(f"Transformed '{censoring_col}' values from [1, 2] to [0, 1].")
    else:
        print(f"'{censoring_col}' already uses [0, 1]. No transformation needed.")

    # Validate the transformation
    print(f"Unique values in '{censoring_col}' after transformation: {dataset[censoring_col].unique()}")

    # Generate the transformed file name with a prefix
    dir_name, file_name = os.path.split(file_path)
    transformed_file_name = f"transformed_{file_name}"
    transformed_file_path = os.path.join(dir_name, transformed_file_name)
    
    # Save the transformed dataset
    dataset.to_csv(transformed_file_path, index=False, sep=";")
    print(f"Transformed dataset saved to '{transformed_file_path}'")
    return transformed_file_path

if __name__ == "__main__":
    # Path to the dataset
    # In the lung cancer dataset, the “time”-field name was changed to "vit_stat_int";
    # and the status-field name was changed to "vit_stat" to preserve the compatability with the syntetic dataset thus requiring less changes in the program
    
    dataset_path = "/Users/narasimha/Downloads/federated-km-main/ncctg_lung_cancer_data.csv"
    data_source = "ncctg_lung_cancer_data"
    
     # Inspect and transform the dataset
    transformed_dataset_path = inspect_and_transform_dataset(dataset_path)

    # Varying number of clients for the experiment
    client_counts = [2, 5, 10, 20, 30, 40, 50]  # List of client counts

    # Run the experiments and collect the computation times
    mean_times, mean_times_he = run_experiments(transformed_dataset_path, data_source, client_counts, num_runs=10)
    
    
    #Save the computation times graph for the lung cancer dataset
    plot_computation_times_comparison(client_counts, mean_times, mean_times_he, save_path="lung_cancer_computation_times.png")
    #plot_computation_times(client_counts, mean_times_he, save_path="lung_cancer_computation_times_he.png")


    dataset_path = "synthetic_breast_data.csv"
    data_source = "synthetic_breast_data"

    transformed_dataset_path = inspect_and_transform_dataset(dataset_path)

    #Run the experiments and collect the computation times for breast cancer dataset
    mean_times, mean_times_he = run_experiments(transformed_dataset_path, data_source, client_counts, num_runs=10)

    #Save the computation times graph for the breast cancer dataset
    plot_computation_times_comparison(client_counts, mean_times, mean_times_he, save_path="breast_cancer_computation_times.png")
