import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import laplace
import matplotlib.pyplot as plt

data_path = "ncctg_lung_cancer_data.csv"
data = pd.read_csv(data_path)
# Define survival time and event observed variables based on dataset structure
time = data['time']
event_observed = (data['status'] == 2).astype(int)
# Compute non-private KM estimate

kmf = KaplanMeierFitter()
kmf.fit(time, event_observed)
survival_times = kmf.survival_function_.index
survival_probs = kmf.survival_function_["KM_estimate"].values

plt.figure(figsize=(10, 8))  # Increase figure size
plt.step(survival_times, survival_probs, where="post", label="Non-Private KM Estimate", color="black")
plt.title("Non-Private Kaplan-Meier Estimate for NCCTG Lung Cancer Data", fontsize=16)  
plt.savefig("non_private_km_plot.pdf")  # Save as PDF
