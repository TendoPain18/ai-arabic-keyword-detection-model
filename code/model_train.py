from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical


class ModelTrain:
    def __init__(self, path, folder_names):
        self.dataset_path = path
        self.folder_names = folder_names

        self.model = 0
        self.history = 0

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

    def load_data_file(self, name):
        feature_sets = np.load(name)
        self.x_train = feature_sets['x_train']
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1)
        self.y_train = feature_sets['y_train']

        self.x_val = feature_sets['x_val']
        self.x_val = self.x_val.reshape(self.x_val.shape[0], self.x_val.shape[1], self.x_val.shape[2], 1)
        self.y_val = feature_sets['y_val']

        self.x_test = feature_sets['x_test']
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1)
        self.y_test = feature_sets['y_test']

    def mark_key_words(self, keywords):
        wake_word_indices = [self.folder_names.index(keyword) for keyword in keywords]

        for i in range(len(wake_word_indices)):
            wake_word_indices[i] = wake_word_indices[i]

        for i in range(len(self.y_train)):
            self.y_train[i] = self.y_train[i]
            if self.y_train[i] in wake_word_indices:
                self.y_train[i] = wake_word_indices.index(self.y_train[i])
            else:
                self.y_train[i] = len(wake_word_indices)

        for i in range(len(self.y_val)):
            self.y_val[i] = self.y_val[i]
            if self.y_val[i] in wake_word_indices:
                self.y_val[i] = wake_word_indices.index(self.y_val[i])
            else:
                self.y_val[i] = len(wake_word_indices)

        for i in range(len(self.y_test)):
            self.y_test[i] = self.y_test[i]
            if self.y_test[i] in wake_word_indices:
                self.y_test[i] = wake_word_indices.index(self.y_test[i])
            else:
                self.y_test[i] = len(wake_word_indices)

        self.y_train = to_categorical(self.y_train, 8)
        self.y_val = to_categorical(self.y_val, 8)
        self.y_test = to_categorical(self.y_test, 8)

        # self.y_train = np.equal(self.y_train, wake_word_indices[0]).astype('float64')
        # self.y_val = np.equal(self.y_val, wake_word_indices[0]).astype('float64')
        # self.y_test = np.equal(self.y_test, wake_word_indices[0]).astype('float64')

    def create_model(self):
        sample_shape = self.x_test.shape[1:]

        self.model = models.Sequential()

        self.model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=sample_shape))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(32, (2, 2), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(8, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])

    def model_summary(self):
        self.model.summary()

    def fit_data(self):
        self.history = self.model.fit(self.x_train,
                                      self.y_train,
                                      epochs=30,
                                      batch_size=100,
                                      validation_data=(self.x_val, self.y_val))

    def plot_test(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    def save_model(self, name):
        models.save_model(self.model, name)
