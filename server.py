# server.py
import numpy as np

class FederatedServer:
    def __init__(self):
        self.global_model = None
        self.client_weights = []

    def aggregate_models(self, local_weights):
        new_weights = list()
        for weights_list_tuple in zip(*local_weights):
            new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
        return new_weights

    def distribute_model(self, clients):
        for client in clients:
            client.set_model_weights(self.global_model)
