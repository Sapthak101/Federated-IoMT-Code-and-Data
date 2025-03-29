# evaluate.py
from dqn_model import build_dqn
import pandas as pd
import tensorflow as tf

test_df = pd.read_csv("data/test.csv")
x_test = test_df.drop('label', axis=1).values
y_test = tf.keras.utils.to_categorical(test_df['label'])

model = build_dqn()
model.load_weights("global_model_weights.h5")
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {acc * 100:.2f}%")
