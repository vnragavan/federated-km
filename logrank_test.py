import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Example data: survival times and event indicators for two groups
# Group 1
data_group1 = pd.DataFrame({
    'time': [5, 6, 6, 2, 4, 8],
    'event': [1, 1, 0, 1, 1, 0]
})

# Group 2
data_group2 = pd.DataFrame({
    'time': [3, 4, 4, 5, 7, 8],
    'event': [1, 0, 1, 1, 0, 1]
})

# Fit Kaplan-Meier curves
kmf_group1 = KaplanMeierFitter()
kmf_group2 = KaplanMeierFitter()

kmf_group1.fit(data_group1['time'], event_observed=data_group1['event'], label="Group 1")
kmf_group2.fit(data_group2['time'], event_observed=data_group2['event'], label="Group 2")

# Plot the survival curves
kmf_group1.plot_survival_function()
kmf_group2.plot_survival_function()

# Perform the log-rank test
results = logrank_test(data_group1['time'], data_group2['time'], 
                       event_observed_A=data_group1['event'], 
                       event_observed_B=data_group2['event'])

# Print the results
print("Log-Rank Test Results:")
print(f"Test Statistic: {results.test_statistic}")
print(f"P-Value: {results.p_value}")

if results.p_value < 0.05:
    print("The difference between the survival curves is statistically significant.")
else:
    print("The difference between the survival curves is not statistically significant.")
