"""
This module is a modification of the KMClient class which represents a client in a federated learning setup
for Kaplan-Meier survival analysis. The class handles the client's dataset, computes Kaplan-Meier 
estimations, and provides event and at-risk counts based on a global timescale.

Modules:
- `KMClient`: Class representing a client in a federated learning setup.
"""

import numpy as np
from lifelines import KaplanMeierFitter
from openfhe import *
from math import log2
#import pdb


class KMClientHE:
    """
    A class to represent a client in a federated learning setup for Kaplan-Meier survival analysis.

    Attributes:
        client_id (int): A unique identifier for the client.
        dataset (pd.DataFrame): The dataset subset assigned to the client.
        global_timescale (list): A global timescale used across all clients
        for Kaplan-Meier estimation.

    Methods:
        get_local_time_points():
            Returns the unique time points (vit_stat_int) from the client's dataset.
        compute_kaplan_meier(global_timescale=None):
            Computes the Kaplan-Meier survival estimation for the client’s data,
            optionally using a global timescale.
        _compute_event_counts_at_risk(kmf, global_timescale):
            Computes event counts and at-risk counts at the global timescale time points
            based on the client's data.
    """

    def __init__(self, client_id, data_split, batch_size):
        """
        Initializes the client with its ID and dataset split.

        Parameters:
            client_id (int): Unique identifier for the client.
            data_split (pd.DataFrame): Data subset for the client.

        Raises:
            ValueError: If required columns 'vit_stat_int' or 'vit_stat' are 
            missing from the dataset.
        """
        self.batch_size = batch_size
        self.client_id = client_id
        self.dataset = data_split
        self.global_timescale = None

        # Check required columns
        required_columns = ["vit_stat_int", "vit_stat"]
        missing_columns = [
            col for col in required_columns if col not in self.dataset.columns
        ]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    def get_local_time_points(self):
        """
        Get unique local time points from the client's dataset.

        Returns:
            list: Local time points.

        Example:
            If the dataset has time points [1, 2, 2, 3], this will return [1, 2, 3].
        """
        return self.dataset["vit_stat_int"].unique().tolist()
    
    def partial_decrypt(self, cipher_text, lead):
        if (lead == True):
            return self.crypto_context.MultipartyDecryptLead([cipher_text], self.key.secretKey)
        x = self.crypto_context.MultipartyDecryptMain([cipher_text], self.key.secretKey)
        return self.crypto_context.MultipartyDecryptMain([cipher_text], self.key.secretKey)
    
    def split_and_encrypt(self, values):
        #values-list is splitted into n chunks, max size of a chunk is batch size, batch_size = 16 for this simulation 
        #This is done due to the limitations imposed by OpenFHE. If the chunk size exceeds 16, an exception with the following message is raised:
        #"CKKSPackedEncoding<lbcrypto::ILDCRTParams<bigintdyn::ubint<long unsigned int> > >(): The number of slots cannot be smaller than value vector size
        ciphertexts = []
        chunks = [values[i:i + self.batch_size] for i in range(0, len(values), self.batch_size)]
        for chunk in chunks:
             plaintext = self.crypto_context.MakeCKKSPackedPlaintext(chunk)
             ciphertexts.append (self.crypto_context.Encrypt(self.public_key, plaintext))
        return ciphertexts

    def compute_kaplan_meier(self, global_timescale=None):
        """
        Computes Kaplan-Meier estimation based on the client's dataset.

        Parameters:
            global_timescale (list, optional): Global timescale for the computation. If None, global timescale is not used.

        Returns:
            tuple: Event counts and at-risk counts for each time point if global_timescale is provided, otherwise None.

        Example:
            Returns event and at-risk counts for each time point if the global_timescale is provided.
        """
        kmf = KaplanMeierFitter()

        if global_timescale is not None:
            # Fit Kaplan-Meier model to the client's data
            kmf.fit(
                self.dataset["vit_stat_int"], event_observed=self.dataset["vit_stat"]
            )
            event_counts, at_risk_counts = self._compute_event_counts_at_risk(
                kmf, global_timescale
            )
            cipher_event_counts = self.split_and_encrypt(event_counts)
            cipher_at_risk_counts  = self.split_and_encrypt(at_risk_counts)
            return cipher_event_counts, cipher_at_risk_counts
            #return event_counts, at_risk_counts

        return None, None

    def _compute_event_counts_at_risk(self, kmf, global_timescale):
        """
        Computes event counts and at-risk counts based on the global timescale.

        Parameters:
            kmf (KaplanMeierFitter): A Kaplan-Meier fitter object that has been fitted to the client's data.
            global_timescale (list): Global timescale for the computation (sorted list of time points).

        Returns:
            tuple: Event counts and at-risk counts for each time point on the global timescale.

        Example:
            If the global_timescale is [1, 2, 3], this function will return the event and at-risk counts for each time point.
        """
        event_counts = []
        at_risk_counts = []

        for time_point in global_timescale:
            # Count individuals at risk at the current time point
            at_risk = np.sum(self.dataset["vit_stat_int"] >= time_point)
            # Count the number of events (deaths) at the current time point
            event = np.sum(
                (self.dataset["vit_stat_int"] == time_point)
                & (self.dataset["vit_stat"] == 1)
            )
            at_risk_counts.append(at_risk)
            event_counts.append(event)

        return event_counts, at_risk_counts
    
    def generate_keys(self, crypto_context, public_key = None):
        #
        self.crypto_context = crypto_context   
        if (public_key == None):
            #leading node
            self.key = self.crypto_context.KeyGen()
        else:
            #next nodes
            self.key = self.crypto_context.MultipartyKeyGen(public_key)
        return self.key.publicKey
    
    def set_public_key(self,pk):
        self.public_key = pk

