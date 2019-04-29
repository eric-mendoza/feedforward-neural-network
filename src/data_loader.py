from PIL import Image
import numpy as np
import random


# This function loads the data set (training, cv, test)
def load_data_image():
    im = Image.open("/Users/Hugo/green_leaves.jpg")
    p = np.array(im)


def load_data():
    # 0) Circle
    data_raw = np.load('../Data/circle.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    circle_input = [np.reshape(x, (784, 1)) for x in data_raw]
    circle_result = [vectorized_result(0) for y in range(7000)]

    # 1) Square
    data_raw = np.load('../Data/square.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    square_input = [np.reshape(x, (784, 1)) for x in data_raw]
    square_result = [vectorized_result(1) for y in range(7000)]

    # 2) Triangle
    data_raw = np.load('../Data/triangle.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    triangle_input = [np.reshape(x, (784, 1)) for x in data_raw]
    triangle_result = [vectorized_result(2) for y in range(7000)]

    # 3) Egg

    # 4) Tree
    data_raw = np.load('../Data/tree.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    tree_input = [np.reshape(x, (784, 1)) for x in data_raw]
    tree_result = [vectorized_result(4) for y in range(7000)]

    # 5) House
    data_raw = np.load('../Data/house.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    house_input = [np.reshape(x, (784, 1)) for x in data_raw]
    house_result = [vectorized_result(5) for y in range(7000)]

    # 6) Happy
    data_raw = np.load('../Data/happy_face.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    happy_input = [np.reshape(x, (784, 1)) for x in data_raw]
    happy_result = [vectorized_result(6) for y in range(7000)]

    # 7) Sad

    # 8) Question

    # 9) Mickey Mouse
    data_mickey = np.load('../Data/mickey.npy')[:7000] / np.float32(255)  # Data between 0 and 1
    mickey_input = [np.reshape(x, (784, 1)) for x in data_mickey]
    mickey_result = [vectorized_result(9) for y in range(7000)]

    # Join all the data
    inputs = np.concatenate((circle_input, square_input, triangle_input, tree_input, house_input, happy_input,
                             mickey_input))
    results = np.concatenate((circle_result, square_result, triangle_result, tree_result, house_result, happy_result,
                              mickey_result))

    join_data = zip(inputs, results)
    join_data = list(join_data)

    # Create the 3 sub datasets
    random.shuffle(join_data)
    train_percent = int(0.80 * len(inputs))
    cv_percent = int(0.10 * len(inputs))

    training = join_data[:train_percent]
    cv = join_data[train_percent: train_percent + cv_percent]
    test = join_data[train_percent + cv_percent:]

    return training, cv, test


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
