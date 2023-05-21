import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class MultiSeriesWindowGenerator():
    def __init__(self, input_width, label_width, shift, batch_size,
                 label_columns=[], regressor_columns=[], static_columns=[],
                 group_by=None, time_column=[]):
        self.batch_size = batch_size
        self.group_by = group_by
        self.time_column = time_column

        self.label_columns = label_columns
        if len(label_columns) != 0:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.regressor_columns = regressor_columns
        self.static_columns = static_columns

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width+shift

        self.input_slice = slice(0, input_width)
        self.input_indicies = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Batch Size: {self.batch_size}',
            f'Label column name(s): {self.label_columns}',
            f'Additional Regressor column name(s): {self.regressor_columns}',
            f'Grouping column(s): {self.group_by}'
        ])
    # take a data set and convert to tensor (n_series, n_batch, n_timestep, n_features)
    def preprocess_data_set(self, data:pd.DataFrame):
        



[DATE]