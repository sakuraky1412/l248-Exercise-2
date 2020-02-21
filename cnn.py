# Python program to create
# Image Classifier using CNN

# Importing the required libraries
import cv2
import os
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import h5py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np

# Setting up the env

train_path = 'Training set'
TEST_DIR = 'Testing set'
fixed_size = (256, 256)
test_size = 0.10
seed = 9
LR = 1e-3
IMG_SIZE = 256
MODEL_NAME = 'scene-{}-{}.model'.format(LR, '6conv-basic')

h5_data = 'Output/multi_data.h5'
h5_labels = 'Output/multi_labels.h5'
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')
gray_featuress_string = h5f_data['dataset_13']
global_labels_string = h5f_label['dataset_1']
trainLabelsGlobal = np.array(global_labels_string)
global_gray_featuress = np.array(gray_featuress_string)
h5f_data.close()
h5f_label.close()
x_train, x_test, y_train, y_test = train_test_split(global_gray_featuress, trainLabelsGlobal, test_size=test_size,
                                                    random_state=seed)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255

# Create the architecture
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 1)))
# Pooling layer with a 2x2 filter to get the max element from the convolved features ,
model.add(MaxPooling2D(pool_size=(2, 2)))
# 2nd Convolution layer with 64 channels
model.add(Conv2D(64, (5, 5), activation='relu'))
# Adding second Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening, Flattens the input. Does not affect the batch size.
model.add(Flatten())
# a layer with 1000 neurons and activation function ReLu
model.add(Dense(1000, activation='relu'))
# a layer with 10 output neurons for each label using softmax activation function
model.add(
    Dense(4, activation='softmax'))
# loss function used for classes that are greater than 2)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot,
                 batch_size=256, epochs=10, validation_split=0.3)

model.evaluate(x_test, y_test_one_hot)[1]
model.save('my_model.h5')
model = load_model('my_model.h5')

train_labels = ['bridge', 'coast', 'mountain', 'rainforest']
test_path = 'Testing set/'
labels = []
preds = []
for label_id, label in enumerate(train_labels):
    if not label.startswith('.'):
        cur_test_path = os.path.join(test_path, label)
        image_names = os.listdir(cur_test_path)
        for img_id, image_name in enumerate(image_names):
            if not image_name.startswith('.'):
                # read the image
                image_path = os.path.join(cur_test_path, image_name)
                image = cv2.imread(image_path)
                # resize the image
                image = cv2.resize(image, fixed_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image.reshape(IMG_SIZE, IMG_SIZE, 1)
                probabilities = model.predict(np.array([image, ]))
                index = np.argsort(probabilities[0, :])
                preds.append(train_labels[index[3]])
                labels.append(label)

print(classification_report(labels, preds, target_names=train_labels))
