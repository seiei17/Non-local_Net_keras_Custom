# model file
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.initializers import he_normal as initial

# from non_local import non_local
from non_local_other import non_local_block as non_local


def bn_relu(input):
    norm = BatchNormalization(axis=3)(input)
    return Activation('relu')(norm)


def bn_relu_conv(filters, kernel_size, strides=(1, 1), padding='same', w_decay=0.0001):
    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_regularizer=l2(w_decay),
                      kernel_initializer=initial())(activation)
    return f


def shortcut(input, residual, w_decay):
    input_shape = K.int_shape(input)
    res_shape = K.int_shape(residual)
    wid_eq = int(round(input_shape[1] / res_shape[1]))
    hei_eq = int(round(input_shape[2] / res_shape[2]))
    cha_eq = input_shape[3] == res_shape[3]
    shortcut = input
    if wid_eq > 1 or hei_eq > 1 or not cha_eq:
        shortcut = Conv2D(res_shape[3], (1, 1),
                          strides=(wid_eq, hei_eq),
                          padding='same',
                          kernel_regularizer=l2(w_decay),
                          kernel_initializer=initial())(input)
    return Add()([shortcut, residual])


def residual_block(filters, w_decay=0.0001, is_first_block=False, is_first_layer=False):
    def f(input):
        strides = (1, 1)
        if is_first_block and not is_first_layer:
            strides = (2, 2)
        if is_first_block and is_first_layer:
            conv1 = Conv2D(filters, (3, 3),
                            strides=strides,
                            padding='same',
                            kernel_regularizer=l2(w_decay))(input)
        else:
            conv1 = bn_relu_conv(filters, (3, 3),
                                 strides=strides,
                                 padding='same',
                                 w_decay=w_decay)(input)
        residual = bn_relu_conv(filters=filters,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                w_decay=w_decay)(conv1)
        input = shortcut(input, residual, w_decay)
        return input
    return f


def non_local_resnet_18(input_shape, num_classes, w_decay=0.0001):
    net = {}
    net['input'] = Input(input_shape)

    net['conv1'] = Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='same',
                          kernel_regularizer=l2(w_decay),
                          kernel_initializer=initial()
                          )(net['input'])
    net['bn1'] = BatchNormalization(axis=3)(net['conv1'])
    net['activate1'] = Activation('relu')(net['bn1'])
    net['maxpool'] = MaxPooling2D((3, 3),
                                  strides=(2, 2),
                                  padding='same')(net['activate1'])

    # net['non1'] = non_local(net['maxpool'])

    net['conv2_1'] = residual_block(64, w_decay, True, True)(net['maxpool'])
    net['conv2_2'] = residual_block(64, w_decay)(net['conv2_1'])

    # net['non2'] = non_local(net['conv2_2'])

    net['conv3_1'] = residual_block(128, w_decay, True)(net['conv2_2'])
    net['conv3_2'] = residual_block(128, w_decay)(net['conv3_1'])

    # net['non3'] = non_local(net['conv3_2'])

    net['conv4_1'] = residual_block(256, w_decay, True)(net['conv3_2'])
    net['non4_1'] = non_local(net['conv4_1'])
    net['conv4_2'] = residual_block(256, w_decay)(net['non4_1'])
    net['non4_2'] = non_local(net['conv4_2'])

    net['conv5_1'] = residual_block(512, w_decay, True)(net['non4_2'])
    net['non5_1'] = non_local(net['conv5_1'])
    net['conv5_2'] = residual_block(512, w_decay)(net['conv5_1'])
    net['non5_2'] = non_local(net['conv5_2'], compression=1)

    net['bn2'] = BatchNormalization(axis=3)(net['non5_2'])
    net['activate2'] = Activation('relu')(net['bn2'])
    shape = K.int_shape(net['activate2'])
    net['avg'] = AveragePooling2D((shape[1], shape[2]), strides=(1, 1))(net['activate2'])
    net['flat'] = Flatten()(net['avg'])
    net['output'] = Dense(num_classes, activation='softmax')(net['flat'])

    model = Model(net['input'], net['output'])
    # model.summary()
    return model


def non_local_resnet_34(input_shape, num_classes, w_decay=0.0001):
    net = {}
    net['input'] = Input(input_shape)

    # layer 1
    net['conv1'] = Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='same')(net['input'])
    net['bn1'] = BatchNormalization(axis=3)(net['conv1'])
    net['activate1'] = Activation('relu')(net['bn1'])
    net['maxpool'] = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net['activate1'])

    # layer 2-7, residual block 1-3
    net['conv2_1'] = residual_block(64, w_decay, True, True)(net['maxpool'])
    net['conv2_2'] = residual_block(64, w_decay)(net['conv2_1'])
    net['conv2_3'] = residual_block(64, w_decay)(net['conv2_2'])

    # layer 8-15, residual block 4-7
    net['conv3_1'] = residual_block(128, w_decay, True)(net['conv2_3'])
    net['conv3_2'] = residual_block(128, w_decay)(net['conv3_1'])
    net['conv3_3'] = residual_block(128, w_decay)(net['conv3_2'])
    net['conv3_4'] = residual_block(128, w_decay)(net['conv3_3'])

    # layer 16-27, residual block 8-13
    net['conv4_1'] = residual_block(256, w_decay, True)(net['conv3_4'])
    net['conv4_2'] = residual_block(256, w_decay)(net['conv4_1'])
    net['conv4_3'] = residual_block(256, w_decay)(net['conv4_2'])
    net['conv4_4'] = residual_block(256, w_decay)(net['conv4_3'])
    net['conv4_5'] = residual_block(256, w_decay)(net['conv4_4'])
    net['conv4_6'] = residual_block(256, w_decay)(net['conv4_5'])

    # layer 28-33, residual block 14-17
    net['conv5_1'] = residual_block(512, w_decay, True)(net['conv4_6'])
    net['conv5_2'] = residual_block(512, w_decay)(net['conv5_1'])
    net['conv5_3'] = residual_block(512, w_decay)(net['conv5_2'])

    # layer 34, fc layer
    net['bn2'] = BatchNormalization(axis=3)(net['conv5_3'])
    net['activate2'] = Activation('relu')(net['bn2'])
    shape = K.int_shape(net['activate2'])
    net['avg'] = AveragePooling2D((shape[1], shape[2]), strides=(1, 1))(net['activate2'])
    net['flat'] = Flatten()(net['avg'])
    net['output'] = Dense(num_classes, activation='softmax')(net['flat'])

    model = Model(net['input'], net['output'])
    # model.summary()
    return model
