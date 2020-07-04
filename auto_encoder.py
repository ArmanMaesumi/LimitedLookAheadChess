import numpy as np
from export_parsed_boards import decompress_board

import time
import keras
from keras.layers import Dense, Activation
from keras.models import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint


class AutoEncoderDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, boards, num_boards, batch_size=32, shuffle=True):
        'Initialization'
        with open(boards, 'r') as file:
            self.boards = file.readlines()

        print(self.boards[0])
        self.batch_size = batch_size
        self.num_boards = num_boards
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_boards / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_boards = []
        for k in indexes:
            batch_boards.append([int(s) for s in self.boards[k].split(',')])

        # Generate data
        X, y = self.__data_generation(batch_boards)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_boards)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, board_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        b = []
        for board in board_batch:
            b.append(decompress_board(board))

        b = np.asarray(b)
        return b, b


class AutoEncoder:

    def __init__(self,
                 model_name,
                 compressed_boards,
                 epochs=200,
                 batch_size=128,
                 activation_func='relu',
                 dropout=0.2):

        np.random.seed(42)
        self.model_filename = model_name + '.model'
        self.model_name = model_name
        self.compressed_boards = compressed_boards
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.dropout = dropout
        self.model = None

    def auto_encoder(self):
        board_input = Input(shape=((64*12) + 7,), name='encoder_input')

        encoded = Dense(512, name='encoder_1')(board_input)
        encoded = Activation(activation=self.activation_func)(encoded)

        encoded = Dense(256, name='encoder_2')(encoded)
        encoded = Activation(activation=self.activation_func)(encoded)

        encoded = Dense(128, name='encoder_3')(encoded)
        encoded = Activation(activation=self.activation_func)(encoded)

        decoded = Dense(256, name='decoder_1')(encoded)
        decoded = Activation(activation=self.activation_func)(decoded)

        decoded = Dense(512, name='decoder_2')(decoded)
        decoded = Activation(activation=self.activation_func)(decoded)

        decoded = Dense((64 * 12) + 7, name='decoded')(decoded)
        decoded = Activation(activation='sigmoid')(decoded)

        self.model = Model(board_input, decoded)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self):
        # 27755973
        # 25055970
        training_gen = AutoEncoderDataGenerator(self.compressed_boards, 25055970, batch_size=self.batch_size)
        val_gen = AutoEncoderDataGenerator('encoder_val(2700000).txt', 2700000, batch_size=self.batch_size)
        self.auto_encoder()

        tensorboard = TensorBoard(log_dir="logs\\{}{}".format(self.model_name, time.time()))
        epoch_path = str(self.model_name) + '-{epoch:02d}.model'

        checkpoint = ModelCheckpoint(epoch_path, period=1)
        self.model.fit_generator(generator=training_gen,
                                 validation_data=val_gen,
                                 use_multiprocessing=True,
                                 epochs=self.epochs,
                                 verbose=True,
                                 callbacks=[tensorboard, checkpoint])


if __name__ == '__main__':
    AE = AutoEncoder('encoder', 'encoder_train(25055973).txt', activation_func='relu', epochs=30)
    AE.train()
