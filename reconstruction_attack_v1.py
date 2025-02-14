import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(file_path, delimiter, time_col, status_col):
    # Load dataset with specified delimiter
    data = pd.read_csv(file_path, delimiter=delimiter)

    # Rename columns to generic names for consistency
    data = data.rename(columns={time_col: "time", status_col: "status"})

    # Convert status to binary: 1=event (e.g., death), 0=censored
    data["status"] = data["status"].apply(lambda x: 1 if x == 1 else 0)

    # Filter out duplicates with status = 0
    filtered_data = data[~data.duplicated(subset=["time", "status"], keep=False) | (data["status"] == 1)]

    print(f"Filtered dataset: {filtered_data.shape[0]} rows remain after removing censored duplicates.")
    return filtered_data

def split_dataset_overlap(data, num_providers, overlap_type="small"):
    """
    Splits the dataset among multiple providers, optionally with overlapping rows.
    overlap_type can be "none", "small", or "large".
    """
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data to ensure randomness
    splits = []

    if overlap_type == "small":
        overlap_fraction = 0.1  # 10% overlap
    elif overlap_type == "large":
        overlap_fraction = 0.5  # 50% overlap
    elif overlap_type == "none":
        overlap_fraction = 0.0  # No overlap
    else:
        raise ValueError("Invalid overlap type. Choose from 'small', 'large', or 'none'.")

    overlap_size = int(len(data) * overlap_fraction)
    unique_size = len(data) - overlap_size

    # Split data into unique and overlapping portions
    unique_data = data.iloc[:unique_size]
    overlapping_data = data.iloc[unique_size:]

    split_size = unique_size // num_providers

    for i in range(num_providers):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_providers - 1 else unique_size
        provider_data = unique_data.iloc[start_idx:end_idx]
        if overlap_fraction > 0:
            provider_data = pd.concat([provider_data, overlapping_data]).reset_index(drop=True)
        splits.append(provider_data)

        # (Optional) You can log the distribution of events and censored rows here
        # num_events = provider_data[provider_data["status"] == 1].shape[0]
        # num_censored = provider_data[provider_data["status"] == 0].shape[0]
        # print(f"Provider {i}: Events = {num_events}, Censored = {num_censored}")

    return splits

def create_splits_for_providers(data, provider_counts, overlap_types):
    """
    Creates splits for multiple provider counts and overlap scenarios.
    Returns a nested dictionary:
      { num_providers: { overlap: [split1, split2, ...] } }
    """
    all_splits = {}
    for num_providers in provider_counts:
        all_splits[num_providers] = {
            overlap: split_dataset_overlap(data, num_providers, overlap)
            for overlap in overlap_types
        }
    return all_splits

def calculate_kaplan_meier(data, global_times):
    """
    Calculates basic Kaplan-Meier statistics:
      - n_at_risk
      - d_t (number of events at each time)
    """
    survival_table = []
    n_at_risk = len(data)
    for time in global_times:
        d_t = data[(data['time'] == time) & (data['status'] == 1)].shape[0]  # Events at time t
        survival_table.append({"time": time, "n_at_risk": n_at_risk, "d_t": d_t})
        # Subtract both censored and event rows at 'time'
        n_at_risk -= data[data['time'] == time].shape[0]
    return pd.DataFrame(survival_table)

def reconstruction_attack(federated_stats, local_stats, attacker_idx):
    """
    Performs a reconstruction attack by subtracting attacker’s known stats
    from the federated stats to infer the "other providers" stats.
    """
    attacker_local_stats = local_stats[attacker_idx]
    reconstructed = []
    for _, row in federated_stats.iterrows():
        time, n_at_risk_fed, d_t_fed = row["time"], row["n_at_risk"], row["d_t"]
        local_row = attacker_local_stats[attacker_local_stats["time"] == time]
        if not local_row.empty:
            n_at_risk_local, d_t_local = local_row.iloc[0]["n_at_risk"], local_row.iloc[0]["d_t"]
        else:
            n_at_risk_local, d_t_local = 0, 0

        n_at_risk_other = n_at_risk_fed - n_at_risk_local
        d_t_other = d_t_fed - d_t_local
        reconstructed.append({"time": time, "n_at_risk_other": n_at_risk_other, "d_t_other": d_t_other})
    return pd.DataFrame(reconstructed)

def evaluate_reconstruction(
    reconstructed_stats, 
    target_stats, 
    accuracy_normalization="range"
):
    """
    Evaluate how well the reconstruction attack recovers the true data,
    returning a dictionary of:
      - mae_n_at_risk, mae_d_t
      - rmse_n_at_risk, rmse_d_t
      - r2_n_at_risk, r2_d_t
      - accuracy_n_at_risk, accuracy_d_t (0–1 metric, based on chosen normalization)

    Parameters:
    -----------
    reconstructed_stats : pd.DataFrame
        Columns: ['time', 'n_at_risk_other', 'd_t_other'] for the attacker’s inferred values.
    target_stats : pd.DataFrame
        Columns: ['time', 'n_at_risk', 'd_t'] for the true values.
    accuracy_normalization : str
        How to normalize MAE for the 0–1 "accuracy" metric. Possible values:
          - "range": use (max - min) of the true values
          - "mean":  use mean of the true values
          - "max":   use max of the true values

    Returns:
    --------
    dict with keys:
      [
        "mae_n_at_risk", "mae_d_t",
        "rmse_n_at_risk", "rmse_d_t",
        "r2_n_at_risk", "r2_d_t",
        "accuracy_n_at_risk", "accuracy_d_t"
      ]
    """
    # 1. Align by time
    common_times = set(reconstructed_stats["time"]).intersection(set(target_stats["time"]))
    aligned_target_stats = target_stats[target_stats["time"].isin(common_times)].reset_index(drop=True)
    aligned_reconstructed_stats = reconstructed_stats[reconstructed_stats["time"].isin(common_times)].reset_index(drop=True)

    # 2. Extract arrays
    y_true_n = aligned_target_stats["n_at_risk"].values
    y_pred_n = aligned_reconstructed_stats["n_at_risk_other"].values
    y_true_d = aligned_target_stats["d_t"].values
    y_pred_d = aligned_reconstructed_stats["d_t_other"].values

    # 3. MAE
    mae_n_at_risk = mean_absolute_error(y_true_n, y_pred_n)
    mae_d_t = mean_absolute_error(y_true_d, y_pred_d)

    # 4. Compute MSE -> RMSE manually (avoiding 'squared=False' for older scikit-learn)
    mse_n_at_risk = mean_squared_error(y_true_n, y_pred_n)
    mse_d_t       = mean_squared_error(y_true_d, y_pred_d)
    rmse_n_at_risk = math.sqrt(mse_n_at_risk)
    rmse_d_t = math.sqrt(mse_d_t)

    # 5. R^2
    # If data is constant, r2 can be 0 or negative. We'll handle that gracefully.
    if len(np.unique(y_true_n)) > 1:
        r2_n_at_risk = r2_score(y_true_n, y_pred_n)
    else:
        r2_n_at_risk = 0.0
    if len(np.unique(y_true_d)) > 1:
        r2_d_t = r2_score(y_true_d, y_pred_d)
    else:
        r2_d_t = 0.0

    # 6. Compute 0–1 accuracy from MAE, based on chosen normalization
    def compute_accuracy(mae_value, true_values):
        if accuracy_normalization == "range":
            val_range = true_values.max() - true_values.min()
            scale = val_range if val_range != 0 else 0
        elif accuracy_normalization == "mean":
            mean_val = true_values.mean()
            scale = mean_val if mean_val != 0 else 0
        else:  # "max" or fallback
            scale = true_values.max()

        if scale > 0:
            return max(0, 1 - (mae_value / scale))
        else:
            return 0.0

    accuracy_n_at_risk = compute_accuracy(mae_n_at_risk, y_true_n)
    accuracy_d_t = compute_accuracy(mae_d_t, y_true_d)

    return {
        "mae_n_at_risk": mae_n_at_risk,
        "mae_d_t": mae_d_t,
        "rmse_n_at_risk": rmse_n_at_risk,
        "rmse_d_t": rmse_d_t,
        "r2_n_at_risk": r2_n_at_risk,
        "r2_d_t": r2_d_t,
        "accuracy_n_at_risk": accuracy_n_at_risk,
        "accuracy_d_t": accuracy_d_t
    }

def plot_reconstruction_accuracy(results, datasets_config, output_dir="plots"):
    """
    Plots and saves reconstruction metrics for n_at_risk and d_t
    across different numbers of providers and overlap scenarios.
    The code now includes:
      - Accuracy
      - RMSE
      - R^2

    :param results: Dictionary of reconstruction results.
    :param datasets_config: List of dictionaries containing dataset info.
    :param output_dir: Directory where plots will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    for config in datasets_config:
        file_path = config["file_path"]
        num_providers = sorted(results[file_path].keys())
        overlaps = ["none", "small", "large"]
        dataset_label = os.path.splitext(os.path.basename(file_path))[0]

        # --- PLOT 1: Accuracy (n_at_risk)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            accuracies_n = [
                results[file_path][providers][overlap]["accuracy_n_at_risk"]
                for providers in num_providers
            ]
            plt.plot(num_providers, accuracies_n, marker="o", label=f"n_at_risk ({overlap.capitalize()} Overlap)")
        plt.title(f"Reconstruction Accuracy (n_at_risk) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("Accuracy (0–1)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_n_at_risk_accuracy.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

        # --- PLOT 2: Accuracy (d_t)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            accuracies_d = [
                results[file_path][providers][overlap]["accuracy_d_t"]
                for providers in num_providers
            ]
            plt.plot(num_providers, accuracies_d, marker="o", label=f"d_t ({overlap.capitalize()} Overlap)")
        plt.title(f"Reconstruction Accuracy (d_t) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("Accuracy (0–1)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_d_t_accuracy.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

        # --- PLOT 3: RMSE (n_at_risk)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            rmse_n = [
                results[file_path][providers][overlap]["rmse_n_at_risk"]
                for providers in num_providers
            ]
            plt.plot(num_providers, rmse_n, marker="s", label=f"n_at_risk ({overlap.capitalize()} Overlap)")
        plt.title(f"RMSE (n_at_risk) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("RMSE (n_at_risk)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_rmse_n_at_risk.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

        # --- PLOT 4: RMSE (d_t)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            rmse_d = [
                results[file_path][providers][overlap]["rmse_d_t"]
                for providers in num_providers
            ]
            plt.plot(num_providers, rmse_d, marker="s", label=f"d_t ({overlap.capitalize()} Overlap)")
        plt.title(f"RMSE (d_t) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("RMSE (d_t)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_rmse_d_t.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

        # --- PLOT 5: R^2 (n_at_risk)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            r2_n = [
                results[file_path][providers][overlap]["r2_n_at_risk"]
                for providers in num_providers
            ]
            plt.plot(num_providers, r2_n, marker="^", label=f"n_at_risk ({overlap.capitalize()} Overlap)")
        plt.title(f"R^2 (n_at_risk) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("R^2 (n_at_risk)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_r2_n_at_risk.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

        # --- PLOT 6: R^2 (d_t)
        plt.figure(figsize=(12, 8))
        for overlap in overlaps:
            r2_d = [
                results[file_path][providers][overlap]["r2_d_t"]
                for providers in num_providers
            ]
            plt.plot(num_providers, r2_d, marker="^", label=f"d_t ({overlap.capitalize()} Overlap)")
        plt.title(f"R^2 (d_t) for {dataset_label}")
        plt.xlabel("Number of Providers")
        plt.ylabel("R^2 (d_t)")
        plt.legend()
        plt.grid()
        plot_name = f"{dataset_label}_r2_d_t.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.show()

def main(datasets_config, accuracy_normalization="range"):
    """
    Main analysis function. 
    :param datasets_config: List of dicts, each with file_path, delimiter, time_col, status_col.
    :param accuracy_normalization: 'range', 'mean', or 'max' for how to compute the 0–1 accuracy.
    """
    results = {}
    for config in datasets_config:
        file_path = config["file_path"]
        delimiter = config["delimiter"]
        time_col = config["time_col"]
        status_col = config["status_col"]

        print(f"\nProcessing dataset: {file_path}")
        filtered_data = load_and_preprocess_data(file_path, delimiter, time_col, status_col)

        provider_counts = [2, 3, 5, 10, 20, 30, 40, 50]
        overlap_types = ["none", "small", "large"]
        all_splits = create_splits_for_providers(filtered_data, provider_counts, overlap_types)

        global_times = sorted(filtered_data["time"].unique())

        # Compute local & federated stats
        all_stats = {}
        for num_providers, splits in all_splits.items():
            all_stats[num_providers] = {}
            for overlap, dp_splits in splits.items():
                local_stats = [calculate_kaplan_meier(dp, global_times) for dp in dp_splits]
                # Summation = naive federated stats
                federated_stats = pd.concat(local_stats).groupby("time").sum().reset_index()
                all_stats[num_providers][overlap] = {
                    "local_stats": local_stats,
                    "federated_stats": federated_stats
                }

        # Perform reconstruction & evaluation
        dataset_results = {}
        for num_providers, stats in all_stats.items():
            print(f"\nReconstruction Evaluation for {num_providers} Providers:")
            provider_results = {}
            for overlap, stat in stats.items():
                # Attack from provider 0's perspective
                reconstructed_stats = reconstruction_attack(stat["federated_stats"], stat["local_stats"], attacker_idx=0)
                
                # Evaluate reconstruction
                metrics = evaluate_reconstruction(
                    reconstructed_stats,
                    stat["local_stats"][1],  # compare to provider 1's data for demonstration
                    accuracy_normalization=accuracy_normalization
                )

                mae_n = metrics["mae_n_at_risk"]
                mae_d = metrics["mae_d_t"]
                rmse_n = metrics["rmse_n_at_risk"]
                rmse_d = metrics["rmse_d_t"]
                r2_n   = metrics["r2_n_at_risk"]
                r2_d   = metrics["r2_d_t"]
                acc_n  = metrics["accuracy_n_at_risk"]
                acc_d  = metrics["accuracy_d_t"]

                print(f"  Overlap: {overlap.capitalize()}:")
                print(f"    MAE (n_at_risk): {mae_n:.2f}, RMSE: {rmse_n:.2f}, R^2: {r2_n:.2f}, Accuracy: {acc_n:.2%}")
                print(f"    MAE (d_t):       {mae_d:.2f}, RMSE: {rmse_d:.2f}, R^2: {r2_d:.2f}, Accuracy: {acc_d:.2%}")
                
                provider_results[overlap] = metrics

            dataset_results[num_providers] = provider_results

        results[file_path] = dataset_results

    # Compare results across datasets
    print("\nComparison of Results Across Datasets:")
    for num_providers in [2, 3, 5, 10, 20, 30, 40, 50]:
        print(f"\nResults for {num_providers} Providers:")
        for overlap in ["none", "small", "large"]:
            for config in datasets_config:
                file_path = config["file_path"]
                metric_vals = results[file_path][num_providers][overlap]
                print(f"  Overlap: {overlap.capitalize()}, Dataset: {file_path}")
                print(f"    MAE (n_at_risk): {metric_vals['mae_n_at_risk']:.2f}, "
                      f"RMSE: {metric_vals['rmse_n_at_risk']:.2f}, "
                      f"R^2: {metric_vals['r2_n_at_risk']:.2f}, "
                      f"Accuracy: {metric_vals['accuracy_n_at_risk']:.2%}")
                print(f"    MAE (d_t):       {metric_vals['mae_d_t']:.2f}, "
                      f"RMSE: {metric_vals['rmse_d_t']:.2f}, "
                      f"R^2: {metric_vals['r2_d_t']:.2f}, "
                      f"Accuracy: {metric_vals['accuracy_d_t']:.2%}")

    # Plot results
    plot_reconstruction_accuracy(results, datasets_config)

if __name__ == "__main__":
    datasets_config = [
        {
            "file_path": "ncctg_lung_cancer_data.csv", 
            "delimiter": ",", 
            "time_col": "time", 
            "status_col": "status"
        },
        {
            "file_path": "synthetic_breast_cancer_data.csv", 
            "delimiter": ";", 
            "time_col": "vit_stat_int", 
            "status_col": "vit_stat"
        }
    ]

    # You can choose 'range', 'mean', or 'max' for accuracy_normalization
    main(datasets_config, accuracy_normalization="range")
