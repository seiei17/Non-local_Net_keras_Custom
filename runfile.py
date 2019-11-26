# runfile
import tensorflow
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from math import ceil
import numpy as np
import os

from Non_local_resnet import nonlocal_resnet
from CifarGenerator import CifarGen

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

num_classes = 10
path = '../../database/cifar{}/'.format(num_classes)
checkpoint_path = './history/checkpoint.h5'
txt_path = './history/accuracy.txt'

resume = False
epochs = 200
batch_size = 32
lr = 1e-3
w_decay = 1e-4

depth = 34
# stage = [3, 4, 6, 3]
stage = None

model = nonlocal_resnet((32, 32, 3,), num_classes, w_decay, depth, stage)
gen = CifarGen(path, batch_size, num_classes)


def lr_reducer(epochs):
    new_lr = lr
    if epochs > 180:
        new_lr *= .5e-3
    elif epochs > 160:
        new_lr *= 1e-3
    elif epochs > 120:
        new_lr *= 1e-2
    elif epochs > 80:
        new_lr *= 1e-1
    return new_lr


lr_reduce_scheduler = LearningRateScheduler(lr_reducer, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_path, 'val_accuracy', 1, True, True)
lr_reduce_Plateau = ReduceLROnPlateau('val_loss', np.sqrt(0.1), 5, 1, min_lr=.5e-6)

if resume:
    model.load_weights(checkpoint_path)

x_train, y_train = gen.train_data()
x_test, y_test = gen.test_data(x_train)
print('train data shape is:', x_train.shape[0])

model.compile(Adam(lr), categorical_crossentropy, ['accuracy'])
history = model.fit_generator(gen.train_gen(x_train, y_train),
                              epochs=epochs,
                              verbose=1,
                              callbacks=[lr_reduce_Plateau, lr_reduce_scheduler, checkpoint],
                              workers=4,
                              validation_data=(x_test, y_test),
                              )

tr_acc = np.array(history.history['accuracy']).reshape((-1, 1))
np.savetxt(txt_path, tr_acc, fmt='%.5f')

scores = model.evaluate(x_test, y_test, verbose=1)
print('test loss: ', scores[0])
print('test accu: ', scores[1])