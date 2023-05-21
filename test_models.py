import tensorflow as tf
from generator import WindowGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a function to compile and fit models made with keras & tf:
def compile_and_fit(model, window, patience = 2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience = patience,
                                                      mode='min')
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = [tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs = MAX_EPOCHS,
                        validation_data = window.val,
                        callbacks = [early_stopping])
    return history

# load weather data
wd = pd.read_csv("Data/weather_data_clean_test.csv")
wd = wd[['time_diff', 'azimuth', 'elevation',
         'windX', 'windY', 'temp', 'rh',
         'rain', 'leaf_wetness']]
train_df = wd.iloc[:1400,:]
val_df = wd.iloc[1400:1800,:]
test_df = wd.iloc[1800:,:]

# set parameters:
CONV_WIDTH = 3
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH-1)

# select window:
# wide window:
wide_window = WindowGenerator(input_width = INPUT_WIDTH, label_width = LABEL_WIDTH, shift = 1,
                              train_df = train_df, val_df = val_df, test_df = test_df,
                              label_columns = ['temp']
                              )
# single step window
window = WindowGenerator(input_width = 1, label_width = 1, shift = 1,
                              train_df = train_df, val_df = val_df, test_df = test_df,
                              label_columns = ['temp']
                              )
# short step window
window = WindowGenerator(input_width = 3, label_width=1, shift = 1,
                                    train_df=train_df, val_df=val_df, test_df=test_df,
                                    label_columns=['temp']
                                    )

# long step window
window  = WindowGenerator(input_width = INPUT_WIDTH, label_width = 1, shift = 1,
                          train_df=train_df, val_df=val_df, test_df=test_df,
                          label_columns=['leaf_wetness']
                          )

print(window)

# linear model:
mod = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

# dense model:
mod = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# flattened dense model (only works for exactly the right window size once compiled)
mod = tf.keras.Sequential([
    # Shape: (time, features) >> (time*features) - 3*9 = 27
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1])
])


# convolutiuonal nural network
mod = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters = 32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units = 32, activation='relu'),
    tf.keras.layers.Dense(units = 1),
])

# LSTM network:
mod = tf.keras.Sequential([
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(units = 1)
])

print('Input shape:', window.example[0].shape)
print('Output shape:', mod(window.example[0]).shape)

MAX_EPOCHS = 40
val_performance = {}
performance = {}

history = compile_and_fit(mod, window)

val_performance['dense'] = mod.evaluate(window.val)
performance['dense'] = mod.evaluate(window.test, verbose=0)
print(val_performance)
print(performance)

#print(wide_window)
#print("Wide window")
#print('Input shape:', wide_window.example[0].shape)
#print('Labels shape:', wide_window.example[1].shape)
#print('Output shape:', mod(wide_window.example[0]).shape)


#wide_window.plot(mod, max_subplots=5)
#window.plot(mod, max_subplots=10)
#plt.show()

#ww_plot = wide_window.plot(model = , plot_col='temp')
#plt.show()

#class Baseline(tf.keras.Model):
#    def __init__(self, label_index = None):
#        super().__init__()
#        self.label_index = label_index

#    def call(self, inputs):
#        if self.label_index is None:
#            return inputs
#        result = inputs[:, :, self.label_index]
#        return result[:, :, tf.newaxis]

#baseline = Baseline(label_index= wide_window.column_indices['temp'])

#baseline.compile(loss = tf.keras.losses.MeanSquaredError(),
#                 metrics = [tf.keras.metrics.MeanAbsoluteError()])



#ww_plot = wide_window.plot(model = baseline, plot_col='temp')
#plt.show()
#val_performance = {}
#performance = {}
#val_performance['Baseline'] = baseline.evaluate(wide_window.val)
#performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

#wide_window.plot(model = baseline, plot_col='temp')

#print(val_performance, performance)
