import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import csv

TRAIN_DIR_1 = 'train/autistic'
TRAIN_DIR_2 = 'train/non_autistic'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'Dectect_Autism'


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR_1)):
        path = os.path.join(TRAIN_DIR_1, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    for img in tqdm(os.listdir(TRAIN_DIR_2)):
        path = os.path.join(TRAIN_DIR_2, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(np.array(img_data))

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def create_label(img_name):
    label = img_name.split('(')
    if label[0] == 'autistic ':
        return np.array([1, 0])
    elif label[0] == 'non_autistic ':
        return np.array([0, 1])


if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy', allow_pickle=True)
    # print(train_data)
else:
    train_data = create_train_data()

if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy', allow_pickle=True)
    # print(test_data)
else:
    test_data = create_test_data()

train = train_data
test = test_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

# X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y_test = [i[1] for i in test]

# tf.reset_default_graph()

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

conv1 = conv_2d(conv_input, 32, 4, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 2, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn_layers, tensorboard_dir='E:/collage/Fourth Year/Neural Networks/Labs/Lab 6/Neural Project',
                    tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

for img in os.listdir(TEST_DIR):
    image = cv2.imread('test/' + img, 0)
    test_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict([test_img])[0]
    row = None
    if prediction[0] > prediction[1]:
        row = [img, 1]
    else:
        row = [img, 0]

    with open('Submit.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# img = cv2.imread('test/291.jpg', 0)
# test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
# prediction = model.predict([test_img])[0]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.imshow(img, cmap='gray')
# print(f"autistic: {prediction[0]}, non_autistic: {prediction[1]}")
# plt.show()
