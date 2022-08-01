import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
from io import BytesIO
import base64

class temp():
    class WindowGen():
        def __init__(self, input_width, label_width, shift, 
                   train_df, val_df, test_df,
                   label_columns=None):
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
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
            return '\n' .join([
                   f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])          

    def plot(self, model=None, plot_col = 'Open', max_subplots=3):
        inputs, labels = self.example
        #plot.figure(figsize=(12,8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            #plot.subplot(max_n, 1, n+1)
            #plot.ylabel(f'{plot_col} [normed]')
            #plot.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            #plot.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                #plot.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
            #if n == 0:
                #plot.legend()
        plot.bar(x = range(len(self.train_df.columns)),
        height=model.layers[0].kernel[:,0].numpy())
        axis = plot.gca()
        axis.set_xticks(range(len(self.train_df.columns)))
        _ = axis.set_xticklabels(self.train_df.columns, rotation=90)
        img = BytesIO()
        plot.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        return plot_url
    WindowGen.plot = plot

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    WindowGen.split_window = split_window

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    WindowGen.make_dataset = make_dataset

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
    WindowGen.train = train
    WindowGen.val = val
    WindowGen.test = test
    WindowGen.example = example
    class Baseline(tf.keras.Model):
        def __init__(self, label_index = None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]  

    def compile_and_fit(model, window, patience=2):
        MAX_EPOCHS = 20
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
        return history

     
