from tensorflow.keras import layers, models
import numpy as np

feature_sets = np.load('all_targets_mfcc_sets.npz')

x_test = feature_sets['x_test']
print(len(x_test))
y_test = feature_sets['y_test']
y_test = np.equal(y_test, 26).astype('float64')

total_1 = 0
for idx, y in enumerate(y_test):
    if y == 1:
        total_1 += 1

model = models.load_model('wake_word_stop_model - Copy.h5')

errors_1_not_detected = 0
errors_0_detected = 0

filename_0 = 'example_0.txt'
filename_1 = 'example_1.txt'
file_0 = open(filename_0, 'w')
file_1 = open(filename_1, 'w')

min_0 = 1
max_0 = 0

min_1 = 1
max_1 = 0

for i in range(0, len(x_test)):
    x = model.predict(np.expand_dims(x_test[i], 0))[0][0]
    print('Answer:', y_test[i], ' Prediction:', x)

    print(i)
    print(i)
    print(i)
    print(i)
    print(i)
    print(i)
    print(i)

    if y_test[i] == 0:
        if x > max_0:
            max_0 = x
        if x < min_0:
            min_0 = x
    else:
        if x > max_1:
            max_1 = x
        if x < min_1:
            min_1 = x

    if x >= 0.5 and y_test[i] == 0:
        errors_0_detected += 1
        file_0.write(f"error 0 detected with {x}\n")
    if x < 0.5 and y_test[i] == 1:
        errors_1_not_detected += 1
        file_1.write(f"error 1 not detected with {x}\n")

file_0.write(f"min_0: {min_0}\n")
file_0.write(f"max_0: {max_0}\n")
file_1.write(f"min_1: {min_1}\n")
file_1.write(f"max_1: {max_1}\n")

file_0.close()
file_1.close()
print(len(x_test))
print(total_1)
print('errors_1_not_detected: ', errors_1_not_detected)
print('errors_0_detected: ', errors_0_detected)

model.evaluate(x=x_test, y=y_test)
