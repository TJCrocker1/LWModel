import numpy as np
import pandas as pd
from UpdatedGenerator import WindowGenerator
import tensorflow as tf

# parse arguments
# TODO: build argument parsing section
TIME_STEP = 0.25
LOOK_BACK = 24
LOOK_FORWARD = 0
INPUT_WIDTH = int(TIME_STEP * LOOK_BACK)
SHIFT = int(TIME_STEP * LOOK_FORWARD)
MAX_EPOCHS = 40
BATCH_SIZE = 32

# get input, label and group col names:
labels = ['leaf_wetness']
inputs = ['lat', 'lon', 'year_sin', 'year_cos', 'rain', 'temp', 'rh', 'windX', 'windY', 'azimuth', 'elevation']
group = 'group'

# load train, val, test into memory
train = pd.read_csv('Data/weather_data_train.csv')[labels + inputs + [group]]
val = pd.read_csv('Data/weather_data_val.csv')[labels + inputs + [group]]
test = pd.read_csv('Data/weather_data_test.csv')[labels + inputs + [group]]

# define functions to fit model
def compile_and_fit(model, window, max_epochs, patience = 2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience = patience,
                                                      mode='min')
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = [tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs = max_epochs,
                        validation_data = window.val,
                        callbacks = [early_stopping])
    return history

# define model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(units = 1)
])

# define window
window = WindowGenerator(input_width=INPUT_WIDTH, label_width=1, shift=SHIFT, batch_size=BATCH_SIZE,
                         train_df = train, val_df = val, test_df = val,
                         label_names=labels, input_names=inputs, group_names= group)

history = compile_and_fit(model, window, max_epochs=MAX_EPOCHS)
val_performance = {}
performance = {}

val_performance['dense'] = model.evaluate(window.val)
performance['dense'] = model.evaluate(window.test, verbose=0)

print(val_performance)
print(performance)