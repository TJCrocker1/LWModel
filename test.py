import pandas as pd
import numpy as np
import tensorflow as tf
from generator import WindowGenerator


#wd = pd.read_csv("Data/weather_data_val.csv")
#x = np.vstack(wd.index).shape[1]
#print(np.vstack(wd.index))
#print(wd)
#print(wd.reset_index())
#print(wd)

#print(wd.columns)#


#label_col = ['leaf_wetness']
#regressor_col = ['year_sin', 'year_cos', 'rain', 'temp', 'rh', 'windX', 'windY',
#                'azimuth', 'elevation']
#static_col = ['lat', 'lon']

#col_types = label_col + regressor_col + static_col
#by = ['group'] + ['date_time']
#print(wd)
#print(wd.set_index(by).unstack(-1))

#print(len(wd['station_id']))

#wd_long = wd.set_index(by).unstack(-1)
#out = tf.stack([wd_long[col] for col in col_types], axis = -1)
#print(out)
#print(col_types)

class WindowGenerator():
    def __init__(self, train_df,
                 input_width, label_width, shift, batch_size = 32,
                 label_names = None, input_names = None):
        # save data:
        self.train_df = train_df

        # find the names & indices of inputs and labels: (if not given default to all, none respectively)
        self.data_names = {name: i for i, name in enumerate(train_df.columns)}
        self.label_names = label_names
        if label_names is not None:
            self.label_names = {name: i for i, name in enumerate(label_names)}
        if input_names is not None:
            self.input_names = {name: i for i, name in enumerate(input_names)}

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
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = True, # ??? shuffels batch not time series sequence
            batch_size = self.batch_size # this is the batch size produced by the data set
        )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)



df = pd.DataFrame({'time': ['1', '2', '3', '1', '2', '3', '2', '3', '4', '5'],
                   'Object': np.concatenate([[i] * 5 for i in [1, 2]]),
                   'Feature1': np.random.randint(10, size=10),
                   'Feature2': np.random.randint(10, size=10),
                   'FeatureY': np.random.randint(0, 2, 10)})

#print(df)

#s = slice(0, 5)
#print(df['FeatureY'][s])


window = WindowGenerator(input_width=4, label_width=1, shift=0, batch_size=4, train_df=df,
                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'])
i = 0
for w in window.train:
    print(w)
    if i == 0: break

#rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
#print(rank_1_tensor)

#x = tf.constant([[1, 2], [2, 9], [3, 8]])
#reshaped = tf.reshape(x, [1, 6])

#print(x.shape)
#print(reshaped.numpy())




#for i in range(0, 4):
#    print(window.train)

#print(window.train)

#dataset = tf.data.Dataset.from_tensor_slices(df)



#for w in dataset.window(3, shift=1, drop_remainder=True):
#  print(list(w.as_numpy_iterator()))

#d = {'group': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#     'time': ['1', '2', '3', '1', '2', '3', '2', '3', '4', '5'],
#     'data1': [5, 3, 4, 7, 8, 2, 4, 5, 1, 1],
#     'data2': [9, 0, 8, 2, 7, 8, 8, 6, 3, 0]}
#df = pd.DataFrame(data=d)

#key = ['group', 'time']
#lab = ['data1', 'data2']

#df1 = df.set_index(key).unstack(-1)

#df2 = tf.stack([df1[l] for l in lab], axis=-1)
#print(df)
#print(df1)
#print(df2)



#print(by)


