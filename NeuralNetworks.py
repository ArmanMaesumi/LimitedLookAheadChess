import numpy as np
import time
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model, Input, load_model
from keras.layers.core import Activation
from keras.layers.core import Dense, Dropout
from keras.layers import concatenate
from keras.initializers import he_normal

from DataGenerator import DataGenerator


class MLP:
    """
    Creates an MLP that predicts side advantage in a given chess position.
    This class compiles a keras model, as well as evaluates pre-existing models.
    To create training data, see stockfish_eval.py
    """

    def __init__(self,
                 model_name,
                 pretrained_weights,
                 epochs=200,
                 batch_size=128,
                 activation_func='relu',
                 dropout=0.2,
                 num_classes=2):

        np.random.seed(42)

        self.model_filename = model_name + '.model'
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.dropout = dropout
        self.num_classes = num_classes
        self.model = None

    def train_with_datagen(self):
        tensorboard = TensorBoard(log_dir="logs\\{}{}".format(self.model_name, time.time()))
        epoch_path = str(self.model_name) + '-{epoch:02d}.model'
        checkpoint = ModelCheckpoint(epoch_path, period=4)
        # training_boards = list(np.load('known_scores(2.7).npy').item().items())
        # val_boards = list(np.load('test_set(166438)_old.npy').item().items())

        training_boards = list(np.load('train(18000000)_new.npy').item().items())
        val_boards = list(np.load('test(2000000)_new.npy').item().items())

        # training_boards = list(np.load('expanded_train(3600000).npy').item().items())
        # val_boards = list(np.load('expanded_test(400000).npy').item().items())

        # training_boards = list(np.load('train_set(5000000)_FICS.npy').item().items())
        # val_boards = list(np.load('test_set(500000)_FICS.npy').item().items())
        if self.num_classes > 0:
            print('Ternary classification')
            training_generator = DataGenerator(training_boards, batch_size=128, categorical=True)
            validation_generator = DataGenerator(val_boards, batch_size=128, categorical=True)
            self.ternary_classifier()
        else:
            training_generator = DataGenerator(training_boards, batch_size=128)
            validation_generator = DataGenerator(val_boards, batch_size=128)
            self.regression_with_encoder()

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=2,
                                 epochs=32,
                                 verbose=True,
                                 callbacks=[tensorboard, checkpoint])

    # Classify White Win / Black Win / Draw states
    def ternary_classifier(self):
        board_input = Input(shape=((64 * 12) + 7,), name='board_input')

        x = Dense(1024)(board_input)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(rate=self.dropout)(x)

        x = Dense(512)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(rate=self.dropout)(x)

        x = Dense(256)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(rate=self.dropout)(x)

        main_output = Dense(3, name='main_output')(x)
        main_output = Activation(activation='softmax')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    # Classify White Win/Black Win states
    def binary_classifier(self):
        board_input = Input(shape=((64 * 12) + 3,), name='board_input')

        x = Dense(1024)(board_input)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(512)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(256)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        main_output = Dense(1, name='main_output')(x)
        main_output = Activation(activation='sigmoid')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def regression(self):
        board_input = Input(shape=((64 * 12) + 7,), name='board_input')

        x = Dense(2048)(board_input)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(2048)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(2048)(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        main_output = Dense(1, name='main_output')(x)

        main_output = Activation(activation='tanh')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mse'])

    def regression_with_encoder(self):
        encoder = load_model('encoder_new_relu-06.model')
        # encoder_model = Model(inputs=encoder.input, outputs=encoder.get_layer('activation_3').output)
        board_input = Input(shape=((64 * 12) + 7,), name='board_input')

        encoded = Dense(512, name='encoder_1', weights=encoder.get_layer('encoder_1').get_weights())(board_input)
        encoded = Activation(activation='relu', name='act1')(encoded)
        
        encoded = Dense(256, name='encoder_2', weights=encoder.get_layer('encoder_2').get_weights())(encoded)
        encoded = Activation(activation='relu', name='act2')(encoded)
        
        encoded = Dense(128, name='encoder_3', weights=encoder.get_layer('encoder_3').get_weights())(encoded)
        encoded = Activation(activation='relu', name='act3')(encoded)

        x = Dense(2048, name='eval_1')(board_input)
        x = Activation(activation=self.activation_func, name='act4')(x)
        x = Dropout(self.dropout, name='drop1')(x)

        x = Dense(2048, name='eval_2')(x)
        x = Activation(activation=self.activation_func, name='act5')(x)
        x = Dropout(self.dropout, name='drop2')(x)

        x = Dense(2048, name='eval_3')(x)
        x = Activation(activation=self.activation_func, name='act6')(x)
        x = Dropout(self.dropout, name='drop3')(x)

        main_output = Dense(1, name='evaluation')(x)
        main_output = Activation(activation='tanh', name='final_output')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mse'])
        print(self.model.summary())

if __name__ == '__main__':
    NN_regression_encoder = MLP('regression_18mill', num_classes=0,
                                activation_func='relu', pretrained_weights=None, dropout=0.25)
    NN_regression_encoder.train_with_datagen()