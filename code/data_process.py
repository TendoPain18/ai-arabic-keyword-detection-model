from os.path import join
import librosa
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features
from playsound import playsound


class DataProcess:
    def __init__(self, path, folder_names, filenames, y):
        self.dataset_path = path
        self.folder_names = folder_names
        self.filenames = filenames
        self.y = y

        # Settings
        self.sample_rate = 8000
        self.num_mfcc = 16
        self.len_mfcc = 16

        # Divided data
        self.filenames_train = []
        self.filenames_val = []
        self.filenames_test = []

        self.y_orig_train = []
        self.y_orig_val = []
        self.y_orig_test = []

        self.final_x_train = []
        self.final_x_val = []
        self.final_x_test = []

        self.final_y_train = []
        self.final_y_val = []
        self.final_y_test = []

    def use_percentage_of_available_data(self, percentage):
        self.filenames = self.filenames[:int(len(self.filenames) * percentage)]

    def divide_data_to_train_val_test(self, use_data_percentage=1, val_set_percentage=0.1, test_set_percentage=0.1):
        self.use_percentage_of_available_data(use_data_percentage)
        val_set_size = int(len(self.filenames) * val_set_percentage)
        test_set_size = int(len(self.filenames) * test_set_percentage)

        self.filenames_val = self.filenames[:val_set_size]
        self.filenames_test = self.filenames[val_set_size:(val_set_size + test_set_size)]
        self.filenames_train = self.filenames[(val_set_size + test_set_size):]

        self.y_orig_val = self.y[:val_set_size]
        self.y_orig_test = self.y[val_set_size:(val_set_size + test_set_size)]
        self.y_orig_train = self.y[(val_set_size + test_set_size):]

    def calc_mfcc(self, path):
        signal, fs = librosa.load(path, sr=self.sample_rate)
        return python_speech_features.base.mfcc(signal,
                                                samplerate=fs,
                                                winlen=0.256,
                                                winstep=0.050,
                                                numcep=self.num_mfcc,
                                                nfilt=26,
                                                nfft=2048,
                                                preemph=0.0,
                                                ceplifter=0,
                                                appendEnergy=False,
                                                winfunc=np.hanning).transpose()

    def test_audio(self, idx):
        path = join(self.dataset_path, self.folder_names[int(self.y_orig_train[idx])], self.filenames_train[idx])
        mfcc = self.calc_mfcc(path)
        print('MFCCs:', mfcc)

        plt.figure()
        plt.imshow(mfcc, cmap='inferno', origin='lower')

        print(self.folder_names[int(self.y_orig_train[idx])])
        playsound(path)
        plt.show()

    def extract_features(self, in_files, in_y):
        out_x = []
        out_y = []
        prob_cnt = 0

        for index, filename in enumerate(in_files):
            path = join(self.dataset_path, self.folder_names[int(in_y[index])], filename)

            if not path.endswith('.wav'):
                continue
            mfccs = self.calc_mfcc(path)

            if mfccs.shape[1] == self.len_mfcc:
                out_x.append(mfccs)
                out_y.append(in_y[index])
            else:
                print('Dropped:', index, mfccs.shape)
                prob_cnt += 1

        return out_x, out_y, prob_cnt

    def clean_data(self):
        self.final_x_train, self.final_y_train, prob = self.extract_features(self.filenames_train, self.y_orig_train)
        print('Removed train percentage:', prob / len(self.y_orig_train))

        self.final_x_val, self.final_y_val, prob = self.extract_features(self.filenames_val, self.y_orig_val)
        print('Removed val percentage:', prob / len(self.y_orig_val))

        self.final_x_test, self.final_y_test, prob = self.extract_features(self.filenames_test, self.y_orig_test)
        print('Removed test percentage:', prob / len(self.y_orig_test))

    def save_data(self, file_name):
        np.savez(file_name,
                 folder_names=self.folder_names,
                 x_train=self.final_x_train,
                 y_train=self.final_y_train,
                 x_val=self.final_x_val,
                 y_val=self.final_y_val,
                 x_test=self.final_x_test,
                 y_test=self.final_y_test)
