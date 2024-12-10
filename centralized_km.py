"""
This module performs Kaplan-Meier survival analysis on a dataset. It loads a dataset,
preprocesses it by checking for missing values and negative time values, and then
computes the Kaplan-Meier survival curve. The result is plotted and saved as an image,
and the survival data is saved to a CSV file.

"""

import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


def preprocess_data(dataset_path):
    """
    Preprocesses the dataset by checking for required columns, missing values,
    negative time values, and ensuring the outcome variable is binary.

    Parameters:
        dataset_path (str): The path to the CSV dataset file.

    Returns:
        pd.DataFrame: The cleaned dataset after preprocessing.

    Raises:
        ValueError: If required columns are missing, or if there are not enough data points.
    """
    # Load the dataset
    data = pd.read_csv(dataset_path, delimiter=";")

    # Verify the required columns
    required_columns = ["vit_stat_int", "vit_stat"]
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    print("Both required columns ('vit_stat_int', 'vit_stat') are present.")

    # Check for missing values
    missing_values = data[required_columns].isnull().sum()
    if missing_values.any():
        print("Missing values detected:")
        print(missing_values)
        # Drop rows with missing values
        data = data.dropna(subset=required_columns)
        print("Missing values have been removed.")
    else:
        print("No missing values detected.")

    # Check for negative time values
    invalid_times = data[data["vit_stat_int"] < 0]
    if len(invalid_times) > 0:
        print(
            f"Found {len(invalid_times)} records with negative time values. Removing them."
        )
        data = data[data["vit_stat_int"] >= 0]
    else:
        print("No negative time values found.")

    # Check if the outcome variable is binary
    unique_outcomes = data["vit_stat"].unique()
    if not set(unique_outcomes).issubset({0, 1}):
        print(
            f"Invalid values in outcome variable 'vit_stat': {unique_outcomes}."
        )
        data = data[data["vit_stat"].isin([0, 1])]
    else:
        print("Outcome variable ('vit_stat') is binary and valid.")

    # Ensure there are enough observations
    print(f"Total observations: {len(data)}")
    event_distribution = data["vit_stat"].value_counts()
    print("Event distribution:")
    print(event_distribution)

    if len(data) < 10:
        raise ValueError("Not enough data to plot a Kaplan-Meier curve.")

    return data


def plot_kaplan_meier_curve(data):
    """
    Fits a Kaplan-Meier estimator to the dataset and plots the survival curve.

    Parameters:
        data (pd.DataFrame): The cleaned dataset containing survival times and events.

    Saves:
        Kaplan-Meier curve plot as 'kaplan_meier.png'.
    """
    # Fit Kaplan-Meier model
    kmf = KaplanMeierFitter()
    kmf.fit(data["vit_stat_int"], event_observed=data["vit_stat"])

    # Extract survival probabilities
    survival_prob = kmf.survival_function_["KM_estimate"]
    time = survival_prob.index

    # Plot Kaplan-Meier curve
    plt.figure(figsize=(10, 6))
    plt.step(
        time,
        survival_prob,
        where="post",
        label="Kaplan-Meier Curve",
        color="blue",
        linewidth=2,
    )
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Curve (Centralized)")
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG image
    output_file = "kaplan_meier.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Kaplan-Meier plot saved as '{output_file}'.")

    # Show the plot
    plt.show()

    # Save the survival function to a CSV file
    survival_data = kmf.survival_function_
    csv_file = "km_survival_data.csv"
    survival_data.to_csv(csv_file, index=False)
    print(f"Kaplan-Meier survival function saved as '{csv_file}'.")
    return survival_data, survival_prob


if __name__ == "__main__":

    # Path to the dataset
    dataset_path = "synthetic_data.csv"

    # Preprocess the dataset
    data = preprocess_data(dataset_path)

    # Plot Kaplan-Meier survival curve
    plot_kaplan_meier_curve(data)
