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
                 label_names = None, input_names = None, group_names = None):
        # save data:
        self.train_df = train_df


        # find the names & indices of inputs and labels: (if not given default to all, none respectively)
        self.data_names = {name: i for i, name in enumerate(train_df.columns)}
        self.label_names = label_names
        if label_names is not None:
            self.label_names = {name: i for i, name in enumerate(label_names)}
        if input_names is not None:
            self.input_names = {name: i for i, name in enumerate(input_names)}
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
            temp_ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data[data[self.group_names] == g],
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,  # shuffels batch not time series sequence
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



df1 = pd.DataFrame({'Time': np.arange(5),
                   'Object': np.concatenate([[i] * 5 for i in [1]]),
                   'Feature1': [1] * 5,
                   'Feature2': [1] * 5,
                   'FeatureY': [1] * 5})

df2 = pd.DataFrame({'Time': np.arange(7),
                   'Object':  np.concatenate([[i] * 7 for i in [2]]),
                   'Feature1': [2] * 7,
                   'Feature2': [2] * 7,
                   'FeatureY': [2] * 7})

data = df1._append(df2)

window = WindowGenerator(input_width=2, label_width=1, shift=0, batch_size=3,
                         train_df = data,
                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'], group_names='Object')

for i in window.train:
    print(i)

#group_names = ['Object']
#gn = {name: i for i, name in enumerate(group_names)}
#print(gn)

#data = np.array(data)

#def afun(data, group_names):
#    return data[group_names].unique()

#print(afun(data, 'Object'))

#group_names = 'Object'
#groups = data[group_names].unique()
#print(groups)
#ds = None
#for g in groups:
#    temp_ds = tf.keras.utils.timeseries_dataset_from_array(
#        data=data[data['Object'] == g],
#        targets=None,
#        sequence_length=2,
#        sequence_stride=1,
#        shuffle=False,  # shuffels batch not time series sequence
#        batch_size=2  # this is the batch size produced by the data set
#    )
#    if ds is not None:
#        ds = ds.concatenate(temp_ds)
#    else:
#        ds = temp_ds

#print(data[data['Object'] == i])

#for i in ds:
#    print(i)


#ds1 = tf.keras.utils.timeseries_dataset_from_array(
#    data=data,
#    targets=None,
#    sequence_length=self.total_window_size,
#    sequence_stride=1,
#    shuffle=True,  # ??? shuffels batch not time series sequence
#    batch_size=self.batch_size  # this is the batch size produced by the data set
#)

#ds = ds1.concatenate(ds2)


#window = WindowGenerator(input_width=2, label_width=1, shift=0, batch_size=1,
#                         train_df1=df1, train_df2=df2,
#                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'])

#for w in window.train:
#  print(w)

#print(df1)
#print(df2)
#df3 = df1._append(df2)
#data3 = np.array(df3)


#dataset = tf.keras.utils.timeseries_dataset_from_array(
#    data = data3,
#    targets = None,
#    sequence_length = 1,
#    sequence_stride = 1,
#    shuffle = True, # ??? shuffels batch not time series sequence
#    batch_size = 1 # this is the batch size produced by the data set
#)

#def dropWindowOverlaps(self, features):



#
#
#dataset.filter(lambda x: x[1] == 2)
#

#for w in dataset:
#  print(w[1])



#ds3_iter = iter(ds3)
#print(next(ds3_iter))
#for i in ds3:
#    print(i)

#data1 = np.array(df1, dtype=np.float32)
#data2 = np.array(df2, dtype=np.float32)

#ds1 = tf.keras.utils.timeseries_dataset_from_array(
#    data = data1,
#    targets = None,
#    sequence_length = 1,
#    sequence_stride = 1,
#    shuffle = True, # ??? shuffels batch not time series sequence
#    batch_size = 3 # this is the batch size produced by the data set
#)

#ds2 = tf.keras.utils.timeseries_dataset_from_array(
#    data = data2,
#    targets = None,
#    sequence_length = 1,
#    sequence_stride = 1,
#    shuffle = True, # ??? shuffels batch not time series sequence
#    batch_size = 3 # this is the batch size produced by the data set
#)

#ds3 = ds1.concatenate(ds2)

#def print_dictionary_dataset(dataset):
#  for i, element in enumerate(dataset):
#    print("Element {}:".format(i))
#    for (feature_name, feature_value) in element.items():
#      print('{:>14} = {}'.format(feature_name, feature_value))

#print_dictionary_dataset(ds3)

#for i in ds3:
#    print(i)



#window1 = WindowGenerator(input_width=4, label_width=1, shift=0, batch_size=4, train_df=df1,
#                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'])
#window2 = WindowGenerator(input_width=4, label_width=1, shift=0, batch_size=4, train_df=df2,
#                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'])

#window3 = window1.concatenate(window1)

#print(type(window1))

#i = 0
#for w in window1.train:
#    print(w[0].numpy())
#    print(type(w[0]))
#    if i == 0: break



#i = 0
#for w in window2.train:
#    print(w)
#    if i == 0: break

#df1 = tf.data.Dataset.from_tensor_slices(
#   (tf.random.uniform([4]),
#    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))#

#print( type(df1) )

#s = slice(0, 5)
#print(df['FeatureY'][s])

#x = np.concatenate([[i] * 5 for i in [1, 2]])

#print([3, 2] * 5)

#window = WindowGenerator(input_width=4, label_width=1, shift=0, batch_size=4, train_df=df,
#                         label_names=['FeatureY'], input_names=['Feature1', 'Feature2'])
#i = 0
#for w in window.train:
#    print(w)
#    if i == 0: break

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


