# client.py
import tensorflow as tf
import pandas as pd
from dqn_model import build_dqn

class IoMTClient:
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.model = build_dqn()
        self.data = pd.read_csv(data_path)

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_model_weights(self):
        return self.model.get_weights()

    def train(self, epochs=1):
        x = self.data.drop('label', axis=1).values
        y = tf.keras.utils.to_categorical(self.data['label'])
        self.model.fit(x, y, epochs=epochs, verbose=0)
