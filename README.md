#  Federated Kaplan-Meier Survival Analysis 

This project simulates federated settings for Kaplan-Meier survival analysis using a dataset. 
The server aggregates survival estimates from multiple clients, computes the Kaplan-Meier curve, 
and saves it as a plot. The survival data is also saved in a CSV file.
The centralized version of KM is also presented for the comparison reasons. 

## Installation

To get started with this project, follow these steps to install the required dependencies.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/kaplan-meier-federated-learning.git
   cd kaplan-meier-federated-learning 
2. **Setup a virtual environment on POSIX (optional but recommended)**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install the required dependencies using requirements.txt**:
   
  ```bash
   pip install -r requirements.txt
```
4. **Install the Python wrapper for OpenFHE** 
   
   Follow the instructions at https://github.com/openfheorg/openfhe-python

5. **Ensure the dataset (synthetic_data.csv) is ready in the root directory of the project**: 

6. **Run the centralized versaion KM.py**:
   ```bash
     python centralized_km.py
   ```
   This will:
   - Load and preprocess the dataset
   - plot the Kaplan-Meier survival curve
   - Save the plot as kaplan_meier.png and the survival function data as km_survival_data.csv
  
7. **Run the federated_simulation for Kaplan-Meier survival analysis**:
    ```bash
     python start_simulation.py
   ```
    This will:
   - split the dataset into client subsets (following uniform random distribution for 2, 5, 10, 20, 30, 40, and 50 clients)
   - Aggregate timepoints and event counts across clients
   - Compute the global Kaplan-Meier curve for each configuration of varying number of clients 2, 5, 10, 20, 30, 40 and 50
   - Save the Kaplan-Meier plot as kaplan_meier_curve_{num_clients}_clients.png.
   - Show and save the computational time plot

8. ***Run the reconstruction attack simulation**:
 ```bash
     python3 reconstruction_attack_v1.py 
   ```
This will:
- split the data into no overlap, small overlap (10%), and large overlap (50%)
- computes the number of individuals at risk and event occurrences over time 
- uses federated statistics to infer missing data from other providers 
- computes MAE, RMSE, Rˆ2 and accuracy for attack effectiveness 
- produces accuracy, RMSE, Rˆ2 graphs across different federated scenarios. 


     
