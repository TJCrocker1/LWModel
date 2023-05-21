# a generator function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#wd = pd.read_csv("Data/weather_data_clean.csv")
#wd = wd[['time_diff', 'azimuth', 'elevation',
#         'windX', 'windY', 'temp', 'rh',
#         'rain', 'leaf_wetness']]
#train_df = wd.iloc[:600000,:]
#val_df = wd.iloc[600000:700000,:]
#test_df = wd.iloc[700000:,:]

#print(wd.columns)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns = None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # find the input column indicies:
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total Window Size: {self.total_window_size}',
            f'Input Indices: {self.input_indices}',
            f'Label Indices: {self.label_indices}',
            f'Label Col Names: {self.label_columns}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = True, # ??? shuffels batch not time series sequence
            batch_size = 32 # this is the batch size produced by the data set
        )
        ds = ds.map(self.split_window)
        return ds

    def plot(self, model=None, plot_col='temp', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

#w1 = WindowGenerator(input_width=24*4, label_width=1, shift=0, label_columns=['leaf_wetness'])
#print( w1.train )

#example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
#                           np.array(train_df[100:100+w1.total_window_size]),
#                           np.array(train_df[200:200+w1.total_window_size])
#                           ])

#for example_labels, example_inputs in w1.train.take(1):
#  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#  print(f'Labels shape (batch, time, features): {example_labels.shape}')



#example_labels, example_inputs = w1.split_window(example_window)

#print('All shapes are: (batch, time, features)')
#print(f'Window shape: {example_window.shape}')
#print(f'Inputs shape: {example_inputs.shape}')
#print(f'Labels shape: {example_labels.shape}')

#print(train_df[200:200+w1.total_window_size])
#def generator(end):
#    value = 0
#    while value < end:
#        yield value
#       value += 1

#for value in generator(4):
#    print(value)
