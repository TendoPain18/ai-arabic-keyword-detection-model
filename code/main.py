from data_collect import *
from data_process import *
from model_train import *

path = 'D:\\Study\\ZC\\projects\\courses\\learning_project\\voice_recognition\\dataset'

dc = DataCollect(path)
dc.remove_folder_name('_background_noise_')
dc.print_folders_names()
dc.print_each_folder_samples_num()
dc.print_total_samples_num()
filenames, y = dc.get_samples()

pr = DataProcess(path, dc.get_folders_names_list(), filenames, y)
pr.divide_data_to_train_val_test(1, 0.1, 0.1)
pr.clean_data()
pr.save_data('all_targets_mfcc_sets.npz')

mt = ModelTrain(path, dc.get_folders_names_list())
mt.load_data_file('all_targets_mfcc_sets.npz')
mt.mark_key_words(['one', 'two', 'three', 'four', 'five', 'on', 'off'])
mt.create_model()
mt.model_summary()
mt.fit_data()
mt.plot_test()
mt.save_model('wake_word_stop_model.h5')

