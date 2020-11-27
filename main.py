import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D,Dropout, BatchNormalization, Input
from tensorflow.keras import Model
import os
import time
from IPython import display
import math

# Load the dataset
data_dir = 'D:\Downloads\Datasets'

dataset, info = tfds.load('celeb_a', with_info=True, data_dir=data_dir)

# Keys needed for the dataset
BATCH_SIZE =32
ATTR_KEY = "attributes"
IMG_KEY = "image"
LABEL_KEY = "Blond_Hair"
GROUP_KEY = "Eyeglasses"
IMG_SIZE =180

# Normalize and resize the image

def augment_data(image):
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_flip_left_right(image)
    return image


def normalize_img(img):
    img = tf.cast(img,tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img / 127.5) - 1
    return img

# Preprocessing the data using the feat dictionary
def preprocess_input(feat_dict):
    image = feat_dict[IMG_KEY]
    label = feat_dict[ATTR_KEY][LABEL_KEY]
    group = feat_dict[ATTR_KEY][GROUP_KEY]

    # Augment,  Resize the images and normalize the image
    image = augment_data(image)
    image = normalize_img(image)

    # Cast the label and group to a float
    label = tf.cast(label, tf.float32)
    group = tf.cast(group, tf.float32)

    feat_dict[IMG_KEY] = image
    feat_dict[ATTR_KEY][LABEL_KEY] = label
    feat_dict[ATTR_KEY][GROUP_KEY] = group

    # Return image with corresponding labels
    return feat_dict[IMG_KEY],feat_dict[ATTR_KEY][LABEL_KEY]

# Return image with group key

train_ds, test_ds, val_ds = dataset['train'],dataset['test'],dataset['validation']

# Preprocessing, shuffling, and batching for the needed datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE
size_of_train_ds = info.splits['train'].num_examples

# Training Dataset
train_ds = train_ds.cache().map(preprocess_input , num_parallel_calls= AUTOTUNE).shuffle(size_of_train_ds).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Test Dataset
test_ds = test_ds.cache().map(preprocess_input, num_parallel_calls = AUTOTUNE).batch(1000).prefetch(AUTOTUNE)

# Validation Set
val_ds = val_ds.cache().map(preprocess_input , num_parallel_calls= AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Show the image relating to the label
def show_images(images):
   for image in images:
    # Retrieving label values
    label_1 = image[1]
    # Show all the image where classes 1 and 2 are present
    if label_1[0] == 1:
        plt.title('Blonde')
        plt.imshow(image[0][0])
        plt.show()
        break


# show_images(train_ds)

# CNN Model
def CNNModel():
    input_img = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(8, (2,2), strides=2, activation='relu')(input_img)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16, (2,2) , strides=2, activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs = input_img, outputs = output)
    return model

# Model Description
# model = CNNModel()
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(train_ds,epochs=1,steps_per_epoch=math.ceil(size_of_train_ds / BATCH_SIZE),verbose=1, validation_data=(val_ds))
# model.save('CNN_Model.h5')

# Load Model
model = tf.keras.models.load_model('CNN_Model.h5')

# Retrieve Sample Batch
sample = next(iter(test_ds))

# Make a prediction on each image in the batch
predict = model.predict(sample[0])

# Show images where the user is likely to be blonde (70%) inside of the batch
for i in range(len(predict)):
    if predict[i] > 0.7:
        plt.imshow(sample[0][i])
        plt.show()