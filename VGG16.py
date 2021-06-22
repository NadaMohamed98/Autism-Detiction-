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

conv1_1 = conv_2d(conv_input, 64, 3, activation='relu')
conv1_2 = conv_2d(conv1_1, 64, 3, activation='relu')
pool1 = max_pool_2d(conv1_2, 2, strides=2)

conv2_1 = conv_2d(pool1, 128, 3, activation='relu')
conv2_2 = conv_2d(conv2_1, 128, 3, activation='relu')
pool2 = max_pool_2d(conv2_2, 2, strides=2)

conv3_1 = conv_2d(pool2, 256, 3, activation='relu')
conv3_2 = conv_2d(conv3_1, 256, 3, activation='relu')
conv3_3 = conv_2d(conv3_2, 256, 3, activation='relu')
pool3 = max_pool_2d(conv3_3, 2, strides=2)

conv4_1 = conv_2d(pool3, 512, 3, activation='relu')
conv4_2 = conv_2d(conv4_1, 512, 3, activation='relu')
conv4_3 = conv_2d(conv4_2, 512, 3, activation='relu')
pool4 = max_pool_2d(conv4_3, 2, strides=2)

conv5_1 = conv_2d(pool4, 512, 3, activation='relu')
conv5_2 = conv_2d(conv5_1, 512, 3, activation='relu')
conv5_3 = conv_2d(conv5_2, 512, 3, activation='relu')
pool5 = max_pool_2d(conv5_3, 2, strides=2)

# full_layer1 = fully_connected(pool5, 1024, activation='relu',)
# full_layer2 = fully_connected(full_layer1, 1024, activation='relu')
# full_layer3 = fully_connected(full_layer2, 1024, activation='relu')
# full_layer3 = dropout(full_layer3, 0.5)
# vggLayers = fully_connected(full_layer3, 4096, activation='relu', scope='fc6')

x = tflearn.fully_connected(pool5, 4096, activation='relu', scope='fc6')
x = tflearn.dropout(x, 0.5, name='dropout1')
x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
x = tflearn.dropout(x, 0.5, name='dropout2')
x = tflearn.fully_connected(x, 2, activation='softmax', scope='fc8')

vggLayers = regression(x, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(vggLayers, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=3,
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

# img = cv2.imread('test/337.jpg', 0)
# test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
# prediction = model.predict([test_img])[0]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.imshow(img, cmap='gray')
# print(f"autistic: {prediction[0]}, non_autistic: {prediction[1]}")
# plt.show()
