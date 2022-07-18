from keras import layers
from keras import losses
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from keras.layers import BatchNormalization

class NetworkModel:
    def __init__(self, context_length, sequence_length, size_encoding, best_model_path, batch_size):
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.size_encoding = size_encoding
        self.best_model_path = best_model_path
        self.batch_size = batch_size

    def train_simple_cnn_model(self, x_train, y_train, x_val, y_val, params):
        layer_input = layers.Input(shape=(2 * self.context_length + self.sequence_length % 2, self.size_encoding))
        layer_conv_1 = layers.Conv1D(filters=params['layer1_units'], kernel_size=3, strides=1, activation="relu", padding="same")(layer_input)
        layer_conv_1 = BatchNormalization()(layer_conv_1)
        layer_conv_1 = layers.Conv1D(filters=params['layer1_units'], kernel_size=3, strides=1, activation="relu", padding="same")(layer_conv_1)
        layer_pooling_1 = layers.AveragePooling1D(pool_size=2)(layer_conv_1)
        layer_pooling_1 = layers.Dropout(params['dropout'])(layer_pooling_1)
        layer_conv_2 = layers.Conv1D(filters=params['layer2_units'], kernel_size=3, strides=1, activation="relu", padding="same")(layer_pooling_1)
        layer_conv_2 = BatchNormalization()(layer_conv_2)
        layer_pooling_2 = layers.AveragePooling1D(pool_size=2)(layer_conv_2)
        layer_pooling_2 = layers.Dropout(params['dropout'])(layer_pooling_2)
        layer_flatten = layers.Flatten(data_format="channels_last")(layer_pooling_2)
        layer_hidden = layers.Dense(units=int(params['hu_dense']), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(layer_flatten)
        layer_output = layers.Dense(units=2, activation='softmax')(layer_hidden)

        model = models.Model(inputs=layer_input, outputs=layer_output)

        from keras.callbacks import EarlyStopping
        from matplotlib import pyplot

        model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
        out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=1500, verbose=1,restore_best_weights=True)], batch_size=self.batch_size, validation_data=(x_val, y_val), verbose=1)
        # # evaluate the model
        # _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        # _, test_acc = model.evaluate(x_val, y_val, verbose=0)
        # print('Train:'+str(train_acc)+'Test: '+ str(test_acc))
        # # plot training history
        # print (out.history)
        # pyplot.plot(out.history['loss'], label='train')
        # pyplot.plot(out.history['val_loss'], label='val')
        # pyplot.legend()
        # pyplot.show()
        return out, model
