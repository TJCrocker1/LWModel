import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class WindowGenerator():
    def __init__(self, train_df, val_df, test_df,
                 input_width, label_width, shift, batch_size=32,
                 label_names=[], input_names=[], group_names=''):

        # save data:
        self.train_df = train_df  #[label_names + input_names + [group_names]]
        self.val_df = val_df
        self.test_df = test_df

        # find the names & indices of inputs and labels: (if not given default to all, none respectively)
        self.data_names = {name: i for i, name in enumerate(train_df.columns)}
        if label_names is not None:
            self.label_names = {name: i for i, name in enumerate(label_names)}
        if input_names is not None:
            self.input_names = {name: i for i, name in enumerate(input_names)}

        # TODO: modify class to cope with none grouped data
        if group_names is not None:
            self.group_names = group_names

        # window parameters:
        self.batch_size = batch_size
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
            f'Label Col Names: {self.label_names}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_names is not None:
            labels = tf.stack(
                [labels[:, :, self.data_names[name]] for name in self.label_names],
                axis=-1
            )
        if self.input_names is not None:
            inputs = tf.stack(
                [inputs[:, :, self.data_names[name]] for name in self.input_names],
                axis=-1
            )
        return inputs, labels

    def make_dataset(self, data):
        groups = data[self.group_names].unique()

        # draw batches from individual groups to avoid overlaps
        ds = None
        for g in groups:
            temp_data = np.array(data[data[self.group_names] == g], dtype=np.float32)
            temp_ds = tf.keras.utils.timeseries_dataset_from_array(
                data= temp_data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,  # shuffels batch not time series sequence
                batch_size=self.batch_size  # this is the batch size produced by the data set
            )
            if ds is not None:
                ds = ds.concatenate(temp_ds)
            else:
                ds = temp_ds

        # slice dataset into inputs and labels:
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)