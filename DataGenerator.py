import numpy as np
import keras
import chess
from export_parsed_boards import parse_FEN


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, boards, batch_size=32, shuffle=True, categorical=False):
        'Initialization'
        self.batch_size = batch_size
        self.boards = boards
        self.shuffle = shuffle
        self.categorical = categorical
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.boards) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_boards = [self.boards[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_boards)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.boards))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, board_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        b = []
        s = []
        # Generate data
        if self.categorical:
            for i, ID in enumerate(board_batch):
                FEN = board_batch[i][0]
                score = board_batch[i][1]
                board = chess.Board(FEN)
                b.append(parse_FEN(board))
                if score > 150:
                    s.append(2)
                elif score < -150:
                    s.append(0)
                else:
                    s.append(1)
        else:
            for i, ID in enumerate(board_batch):
                # Store sample

                FEN = board_batch[i][0]
                score = board_batch[i][1]

                if score > 5000:
                    score = 5000
                elif score < -5000:
                    score = -5000

                b.append(parse_FEN(chess.Board(FEN)))
                s.append(2*((score - (-5000)) / (5000 - (-5000)))-1)

        b = np.asarray(b)
        s = np.asarray(s)
        return b, s