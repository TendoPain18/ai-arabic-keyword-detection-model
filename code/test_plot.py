import matplotlib.pyplot as plt

# Data
strings = ['def', 'abc', 'ghi']
nums = [2, 4, 6]

# Sort data based on the strings (with 'abc' first)
sorted_indices = sorted(range(len(strings)), key=lambda k: strings[k])
sorted_strings = [strings[i] for i in sorted_indices]
sorted_nums = [nums[i] for i in sorted_indices]



plt.scatter(sorted_nums, sorted_strings, color='b', marker='o', s=100)


plt.title('Iteration vs. Predicted')
plt.xlabel('Iteration Number')
plt.ylabel('Predicted Word')
plt.show()


def plot_word_prediction(error, name):
    x_axis = error[0]
    y_axis = error[1]
    plt.scatter(sorted_nums, sorted_strings, color='b', marker='o', s=100)
    plt.title(name)
    plt.xlabel('Iteration Number')
    plt.ylabel('Predicted Word')
    plt.show()
