import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
from keras.utils import to_categorical
from collections import Counter

feature_sets = np.load('all_targets_mfcc_sets.npz')
folder_names = feature_sets['folder_names'].tolist()

keywords = ['one', 'two', 'three', 'four', 'five', 'on', 'off']
wake_word_indices = [folder_names.index(keyword) for keyword in keywords]


def get_total_word_numbers_in_the_dataset(y_test):
    nums_of_occurrence = []

    for i in range(len(wake_word_indices) + 1):
        n = y_test.tolist().count(i)
        nums_of_occurrence.append(n)

    return nums_of_occurrence


def get_test_data():
    x_test = feature_sets['x_test']
    y_test = feature_sets['y_test']
    for i in range(len(y_test)):
        y_test[i] = y_test[i]
        if y_test[i] in wake_word_indices:
            y_test[i] = wake_word_indices.index(y_test[i])
        else:
            y_test[i] = len(wake_word_indices)
    nums = get_total_word_numbers_in_the_dataset(y_test)
    y_test = to_categorical(y_test, 8)
    return x_test, y_test, nums

def plot_words_prediction(errors, keywords):
    for i in range(len(errors)):
        x_axis = errors[i][0]
        y_axis = errors[i][1]
        colors = ['r' if x < 95 else 'g' for x in errors[i][2]]
        plt.figure()
        plt.scatter(x_axis, y_axis, color=colors, marker='o', s=100)
        plt.title('nothing' if i == 7 else keywords[i])
        plt.xlabel('Iteration Number')
        plt.ylabel('Predicted Word')
    plt.show()



x_test, y_test, nums = get_test_data()

not_detected_error_list = [0, 0, 0, 0, 0, 0, 0, 0]
wrong_detect_error_list = [0, 0, 0, 0, 0, 0, 0, 0]
minimums = [1, 1, 1, 1, 1, 1, 1, 1]
maximums = [0, 0, 0, 0, 0, 0, 0, 0]

files = [open('log\\' + file + '.txt', 'w') for file in keywords]

errors = []

for i in keywords:
    errors.append([[], [], []])
errors.append([[], [], []])

model = models.load_model('all_data_7_commands.h5')
iterations = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(0, len(x_test)):
    # x = model.predict(np.expand_dims(x_test[i], 0))[0][0]
    prediction = model.predict(np.expand_dims(x_test[i], 0))
    predicted_word_index = np.argmax(prediction, axis=1)[0]
    prediction_percentage = prediction[0][predicted_word_index] * 100
    predicted_word = 'nothing' if predicted_word_index == 7 else keywords[predicted_word_index]
    t = y_test[i].tolist().index(1)
    original_word = 'nothing' if t == 7 else keywords[t]

    errors[t][0].append(iterations[t])
    errors[t][1].append(predicted_word)
    errors[t][2].append(prediction_percentage)
    iterations[t] += 1

    if predicted_word != original_word and prediction_percentage > 98:
        not_detected_error_list[t] += 1
        wrong_detect_error_list[predicted_word_index] += 1



    print(i)
    print(i)
    print(i)
    print(i)
    print(i)
    print(i)
    print(i)

print(nums)
print(not_detected_error_list)
print(wrong_detect_error_list)

plot_words_prediction(errors, keywords)

#     if prediction[0][predicted_word_index] > 0.5:
#         predicted_word_index = 7
#
#
#     if predicted_word_index:
#         print()
#
#     if y_test[i] == 0:
#         if x > max_0:
#             max_0 = x
#         if x < min_0:
#             min_0 = x
#     else:
#         if x > max_1:
#             max_1 = x
#         if x < min_1:
#             min_1 = x
#
#     if x >= 0.5 and y_test[i] == 0:
#         errors_0_detected += 1
#         file_0.write(f"error 0 detected with {x}\n")
#     if x < 0.5 and y_test[i] == 1:
#         errors_1_not_detected += 1
#         file_1.write(f"error 1 not detected with {x}\n")
#
#     print('Answer:', y_test[i], ' Prediction:', wake_word_indices[predicted_word_index])
#     print(i)
#     print(i)
#     print(i)
#     print(i)
#     print(i)
#     print(i)
#     print(i)
#
#
#
# file_0.write(f"min_0: {min_0}\n")
# file_0.write(f"max_0: {max_0}\n")
# file_1.write(f"min_1: {min_1}\n")
# file_1.write(f"max_1: {max_1}\n")
#
# file_0.close()
# file_1.close()
# print(len(x_test))
# print(total_1)
# print('errors_1_not_detected: ', errors_1_not_detected)
# print('errors_0_detected: ', errors_0_detected)
#
# model.evaluate(x=x_test, y=y_test)
