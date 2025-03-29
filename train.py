# train.py
from client import IoMTClient
from server import FederatedServer
import os

data_dir = 'data'
clients = [IoMTClient(i, os.path.join(data_dir, 'train.csv')) for i in range(5)]
server = FederatedServer()

# Initial global model
server.global_model = clients[0].get_model_weights()

# Federated rounds
for rnd in range(5):
    local_weights = []
    for client in clients:
        client.set_model_weights(server.global_model)
        client.train(epochs=1)
        local_weights.append(client.get_model_weights())
    server.global_model = server.aggregate_models(local_weights)
print("Training completed.")
