"""
This module is a modification of the KMServer class, which is used to aggregate Kaplan-Meier survival estimates 
from multiple clients in a federated learning setup. The server computes the Kaplan-Meier curve 
using event counts and at-risk counts from the clients and saves the curve as an image.

Modules:
    KMServer: A class for federated learning-based Kaplan-Meier survival analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from key_management import KeyManagement


class KMServerHE:
    """
    A class to represent the server in a federated learning setup for 
    Kaplan-Meier survival analysis.

    Attributes:
        num_clients (int): The number of clients in the federated learning setup.
        global_timescale (list): The global time points shared across all clients 
        for Kaplan-Meier estimation.

    Methods:
        aggregate_round_1(clients):
            Aggregates local time points from all clients to generate a global timescale.
        aggregate_round_2(clients):
            Aggregates event and at-risk counts from all clients and computes 
            the Kaplan-Meier curve.
        _compute_survival_probabilities(event_counts, at_risk_counts):
            Computes survival probabilities based on event counts and at-risk counts.
        _save_kaplan_meier_curve(survival_probabilities):
            Saves the Kaplan-Meier curve as an image file.
        generate_keys : 
            Runs distributed key generation process
            sets initial public key equals None
            send pk to clients in a loop 
            recieves updated public key, which is then sent to next client
            final public key equals the public key received from the last client in the loop
            pls note: secret key shares never leave the clients and are known only to their owners (clients that generated these keys)
    """

    def __init__(self, num_clients, clients, batch_size, data_source):
        """
        Initializes the server with the number of clients.

        Parameters:
            num_clients (int): Number of clients in the federated learning setup.
        """
        self.num_clients = num_clients
        self.clients = clients
        self.global_timescale = None
        self.batch_size = batch_size
        self.data_source = data_source

    def generate_crypto_context(self):
        k_m = KeyManagement()
        self.crypto_context = k_m.crypto_context

    def generate_keys(self):
        pk = None
        for client in self.clients:
            pk = client.generate_keys(self.crypto_context, pk)
        for client in self.clients:
            client.set_public_key(pk)
        self.public_key = pk
        
    def aggregate_round_1(self):
        """
        Aggregates unique time points from all clients to create the global timescale.

        Parameters:
            clients (list): List of client objects that will contribute their time points.

        Returns:
            list: The global timescale (sorted set of unique time points across all clients).

        Example:
            If clients have time points [1, 2, 2, 3], the global timescale will be [1, 2, 3].
        """
        all_time_points = []
        for client in self.clients:
            local_time_points = client.get_local_time_points()
            all_time_points.extend(local_time_points)

        self.global_timescale = sorted(set(all_time_points))
        return self.global_timescale
    
    def decryption_round(self, ciphertexts):
        #
        decrypted = []
        aggregated = []
        for ciphertext in ciphertexts:
            lead = True
            partial_decrypts = []
            for client in self.clients:
                partial_decrypt = client.partial_decrypt(ciphertext, lead)
                partial_decrypts.append(partial_decrypt[0])
                lead = False
            decrypted = self.crypto_context.MultipartyDecryptFusion(partial_decrypts)
            decrypted.SetLength(self.batch_size)
            decrypted = decrypted.GetCKKSPackedValue()
            aggregated = aggregated + decrypted
        return aggregated

    def add_ciphertexts(self, ciphertext1, ciphertext2):
        ciphertextAdd = []
        for i in range(len(ciphertext1)):
            ciphertextAdd.append(self.crypto_context.EvalAdd(ciphertext1[i], ciphertext2[i]))
        return ciphertextAdd

    def aggregate_round_2_HE(self):
        """
        Aggregates event counts and at-risk counts from all clients, and computes 
        the global Kaplan-Meier curve.

        Parameters:
            clients (list): List of client objects that will provide their event and at-risk counts.

        Raises:
            ValueError: If the global timescale is not set (i.e., round 1 has not been completed).

        Example:
            This method aggregates data from all clients, computes survival probabilities,
            and saves the Kaplan-Meier curve.
        """
        if self.global_timescale is None:
            raise ValueError("Global timescale must be set in Round 1.")

        aggregated_event_counts = np.zeros(len(self.global_timescale))
        aggregated_at_risk_counts = np.zeros(len(self.global_timescale))
        ciphersum_event_counts = None
        ciphersum_at_risk_counts = None

        for client in self.clients:
            event_counts, at_risk_counts = client.compute_kaplan_meier(
                self.global_timescale
            )
            if (ciphersum_event_counts == None):
                ciphersum_event_counts = event_counts
                ciphersum_at_risk_counts = at_risk_counts
            else:
                ciphersum_event_counts = self.add_ciphertexts(ciphersum_event_counts, event_counts)
                ciphersum_at_risk_counts = self.add_ciphertexts(ciphersum_at_risk_counts, at_risk_counts)

        aggregated_event_counts_complex = np.array(self.decryption_round(ciphersum_event_counts),dtype=complex) 
        aggregated_at_risk_counts_complex  = np.array(self.decryption_round(ciphersum_at_risk_counts),dtype=complex) 
        aggregated_event_counts = aggregated_event_counts_complex.real
        aggregated_at_risk_counts = aggregated_at_risk_counts_complex.real
        survival_probabilities = self._compute_survival_probabilities(
            aggregated_event_counts, aggregated_at_risk_counts
        )
        self._save_kaplan_meier_curve(survival_probabilities)
        return survival_probabilities 

    def _compute_survival_probabilities(self, event_counts, at_risk_counts):
        """
        Computes Kaplan-Meier survival probabilities from event and at-risk counts.

        Parameters:
            event_counts (list): The number of events (deaths) observed at each time point.
            at_risk_counts (list): The number of individuals at risk at each time point.

        Returns:
            list: The Kaplan-Meier survival probabilities at each time point.

        Example:
            Given event counts and at-risk counts, this method computes the survival probabilities
            using the Kaplan-Meier formula.
        """
        survival_probabilities = []
        survival_prob = 1.0
        for i in range(len(self.global_timescale)):
            if at_risk_counts[i] > 0:
                hazard = event_counts[i] / at_risk_counts[i]
                survival_prob *= 1 - hazard
            survival_probabilities.append(survival_prob)
        return survival_probabilities

    def _save_kaplan_meier_curve(self, survival_probabilities):
        """
        Saves the Kaplan-Meier curve as an image file.

        Parameters:
            survival_probabilities (list): The survival probabilities at each time point.

        Example:
            This method generates and saves a plot of the Kaplan-Meier survival curve.
        """
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"kaplan_meier_curve_{self.data_source}_{self.num_clients}_clients_he.png"
        )

        plt.figure(figsize=(10, 6))
        plt.step(
            self.global_timescale,
            survival_probabilities,
            where="post",
            label="Kaplan-Meier Curve",
        )
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title(f"Kaplan-Meier Curve for {self.num_clients} Clients")
        plt.grid()
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Kaplan-Meier curve saved at {output_path}")
