# runfile
import tensorflow
import keras as keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import numpy as np
from math import ceil
import os

from CifarGenerator import CifarGen
from Non_local_resnet import non_local_resnet_18

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# file params
num_classes=100
dataname = 'cifar{}'.format(num_classes)
path = '../../database/{}/'.format(dataname)
checkpoint_path = './history/CheckPoint.h5'
history_path = './history/accuracy.txt'

input_shape = (32, 32, 3,)

# training params
resume = False
epochs = 150
val_size = 0.1
batch_size = 128
lr = 0.001
w_decay = 0.0001

train_steps = ceil(50000 * (1 - val_size) / batch_size)
valid_steps = ceil(5000 * val_size / batch_size)

gen = CifarGen(path, batch_size, num_classes)
model = non_local_resnet_18(input_shape, num_classes, w_decay)
model_checkpoint = ModelCheckpoint(checkpoint_path, 'val_accuracy',
                                   1, True, True)
lr_reduce = ReduceLROnPlateau('val_acc', factor=0.1,
                              patience=10, verbose=1)
callbacls = [lr_reduce, model_checkpoint]

if resume:
    model.load_weights(checkpoint_path)

opt = keras.optimizers.Adam(lr)

x_train, y_train = gen.train_data()
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=val_size)
print('train data shape is:', X_train.shape[0])
print('validation data shape is:', X_val.shape[0])

model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(gen.train_gen(X_train, Y_train),
                              steps_per_epoch=train_steps,
                              epochs=epochs,
                              verbose=1,
                              callbacks=callbacls,
                              validation_data=gen.valid_gen(X_val, Y_val),
                              validation_steps=valid_steps)

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']

np_train_acc = np.array(train_acc).reshape((-1, 1))
np_valid_acc = np.array(valid_acc).reshape((-1, 1))
np_out = np.concatenate([np_train_acc, np_valid_acc], axis=1)
np.savetxt(history_path, np_out, fmt='%.5f')