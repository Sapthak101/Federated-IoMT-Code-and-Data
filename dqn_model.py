# dqn_model.py
import tensorflow as tf

def build_dqn(input_dim=8, output_dim=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
