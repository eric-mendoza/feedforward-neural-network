import numpy as np
import random
from skimage import color
from skimage import io
import qimage2ndarray as converter


# This function loads the data set (training, cv, test)
def load_data_image():
    img = 1 - color.rgb2gray(io.imread('../Generated/1.bmp'))
    return img


def qimage_to_ndarray(img):
    image = converter.rgb_view(img)
    image = 1 - color.rgb2gray(image)
    image = np.reshape(image, (784, 1))
    return image


def image_classifier(i):
    result = ""
    if i == 0:
        result = "Circle"

    elif i == 1:
        result = "Square"

    elif i == 2:
        result = "Triangle"

    elif i == 3:
        result = "Egg"

    elif i == 4:
        result = "Tree"

    elif i == 5:
        result = "House"

    elif i == 6:
        result = "Happy Face"

    elif i == 7:
        result = "Sad Face"

    elif i == 8:
        result = "Question"

    elif i == 9:
        result = "Mickey Mouse"

    return result


def load_data(dataset_size):
    # 0) Circle
    data_raw = np.load('../Data/circle.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    circle_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    circle_result = [vectorized_result(0) for y in range(dataset_size)]

    # 1) Square
    data_raw = np.load('../Data/square.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    square_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    square_result = [vectorized_result(1) for y in range(dataset_size)]

    # 2) Triangle
    data_raw = np.load('../Data/triangle.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    triangle_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    triangle_result = [vectorized_result(2) for y in range(dataset_size)]

    # 3) Egg
    data_raw = np.load('../Data/egg.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    egg_input = [np.reshape(x, (784, 1)) for x in data_raw[:3000]]
    egg_result = [vectorized_result(3) for y in range(3000)]

    # 4) Tree
    data_raw = np.load('../Data/tree.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    tree_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    tree_result = [vectorized_result(4) for y in range(dataset_size)]

    # 5) House
    data_raw = np.load('../Data/house.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    house_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    house_result = [vectorized_result(5) for y in range(dataset_size)]

    # 6) Happy
    data_raw = np.load('../Data/happy_face.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    happy_input = [np.reshape(x, (784, 1)) for x in data_raw[:dataset_size]]
    happy_result = [vectorized_result(6) for y in range(dataset_size)]

    # 7) Sad
    data_raw = np.load('../Data/sad.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    sad_input = [np.reshape(x, (784, 1)) for x in data_raw[:3500]]
    sad_result = [vectorized_result(7) for y in range(3500)]

    # 8) Question
    data_raw = np.load('../Data/question.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    question_input = [np.reshape(x, (784, 1)) for x in data_raw[:3000]]
    question_result = [vectorized_result(8) for y in range(3000)]

    # 9) Mickey Mouse
    data_raw = np.load('../Data/mickey2.npy', mmap_mode='r') / np.float32(255)  # Data between 0 and 1
    mickey_input = [np.reshape(x, (784, 1)) for x in data_raw[:2480]]
    mickey_result = [vectorized_result(9) for y in range(2480)]

    # Join all the data
    inputs = np.concatenate((circle_input, square_input, triangle_input, egg_input, tree_input, house_input,
                             happy_input, sad_input, question_input, mickey_input))
    results = np.concatenate((circle_result, square_result, triangle_result, egg_result, tree_result, house_result,
                              happy_result, sad_result, question_result, mickey_result))

    # Convert image to black and white
    inputs[:] = inputs[:] > 0

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

