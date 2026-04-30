from os import listdir
from os.path import isdir, join
import random
import numpy as np


class DataCollect:
    def __init__(self, path):
        self.dataset_path = path
        self.folder_names = [name for name in listdir(self.dataset_path) if isdir(join(self.dataset_path, name))]
        self.filenames = []
        self.y = []

    def get_folders_names_list(self):
        return self.folder_names

    def remove_folder_name(self, folder_name):
        self.folder_names.remove(folder_name)

    def get_folder_samples_num(self, folder_name):
        return len(listdir(join(self.dataset_path, folder_name)))

    def get_total_samples_num(self):
        samples_num = 0
        for folder_name in self.folder_names:
            samples_num += self.get_folder_samples_num(folder_name)
        return samples_num

    def load_samples(self):
        for index, folder_name in enumerate(self.folder_names):
            contained_files = listdir(join(self.dataset_path, folder_name))
            self.filenames.append(contained_files)
            y_array = np.ones(len(self.filenames[index])) * index
            self.y.append(y_array)

        self.filenames = [item for sublist in self.filenames for item in sublist]
        self.y = [item for sublist in self.y for item in sublist]

    def shuffle_samples(self):
        filenames_y = list(zip(self.filenames, self.y))
        random.shuffle(filenames_y)
        self.filenames, self.y = zip(*filenames_y)

    def get_samples(self):
        self.load_samples()
        self.shuffle_samples()
        return self.filenames, self.y

    #########################################################
    # print functions #
    #########################################################
    def print_folders_names(self):
        for folder_name in self.folder_names:
            if isdir(join(self.dataset_path, folder_name)):
                print(folder_name)

    def print_each_folder_samples_num(self):
        for folder_name in self.folder_names:
            print(folder_name, " folder contain ", self.get_folder_samples_num(folder_name), "files")

    def print_total_samples_num(self):
        print('Total samples: ', self.get_total_samples_num())
