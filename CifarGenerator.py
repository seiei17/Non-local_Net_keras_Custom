# Cifar data file.
# Cifar Generator file.
import pickle
import os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        return data


class CifarGen:
    def __init__(self, path, batch, type=10):
        self.path = path
        self.type = type
        self.batch = batch

    def train_data(self):
        # cifar 10
        global labels, images
        if self.type == 10:
            for i in range(5):
                path = os.path.join(self.path, 'data_batch_{}'.format(i + 1))
                data = load(path)
                if i == 0:
                    images = data['data']
                    labels = data['labels']
                else:
                    image = data['data']
                    label = data['labels']
                    images = np.concatenate([images, image], axis=0)
                    labels = np.concatenate([labels, label], axis=0)
            images = images.reshape(len(labels), 32, 32, 3)
            mean = np.mean(images, axis=0)
            for j in range(len(labels)):
                images[j] = images[j] - mean
            images = images / 128.
            labels = to_categorical(labels, self.type)
            return images, labels
        # cifar 100
        if self.type == 100:
            path = os.path.join(self.path, 'train')
            data = load(path)
            images = data['data']
            labels = data['fine_labels']
            images = images.reshape(len(labels), 32, 32, 3)
            mean = np.mean(images, axis=0)
            for j in range(len(labels)):
                images[j] = images[j] - mean
            images = images / 128.
            labels = to_categorical(labels, self.type)
            return images, labels

    def train_gen(self, x_train, y_train):
        gen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 rotation_range=0,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.05,
                                 zoom_range=0.05,
                                 fill_mode='nearest',
                                 horizontal_flip=True,
                                 vertical_flip=False)
        return gen.flow(x_train, y_train, self.batch)

    def valid_gen(self, x_val, y_val):
        gen = ImageDataGenerator()
        return gen.flow(x_val, y_val, self.batch)

    def test_gen(self, x_test, y_test):
        # cifar 10
        if self.type == 10:
            path = os.path.join(self.path, 'test_batch')
            data = load(path)
            labels = data['labels']
        # cifar 100
        else:
            path = os.path.join(self.path, 'test')
            data = load(path)
            labels = data['fine_labels']
        images = data['data']
        images = images.reshape(len(labels), 32, 32, 3)
        mean = np.mean(images, axis=0)
        for j in range(len(labels)):
            images[j] = images[j] - mean
        images = images / 128.
        labels = to_categorical(labels, self.type)
        gen = ImageDataGenerator()
        return gen.flow(images, labels, self.batch)