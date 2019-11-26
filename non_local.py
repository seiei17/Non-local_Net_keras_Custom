# non-local block file
import keras.backend as K
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Dot
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import MaxPool1D
from keras.layers import Concatenate
from keras.layers import Add

from keras.initializers import he_normal as initial

from math import ceil


def non_local(input, bottle_dim=None, compression=2,
              mode='embedded', residual=True):
    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('mode is not right.')

    shape = K.int_shape(input)
    length, dim1, dim2, channel = shape

    # define bottle_dim
    if bottle_dim is None:
        bottle_dim = ceil(shape[3]) // 2
    else:
        bottle_dim = int(bottle_dim)
        if bottle_dim < 1:
            raise ValueError('bottle_dim must > 1')

    # compute f
    if mode == 'gaussian':
        xi = Reshape((-1, bottle_dim))(input)
        xj = Reshape((-1, bottle_dim))(input)
        f = Dot(axes=2)([xi, xj])
        f = Activation('softmax')(f)

    if mode == 'embedded':
        theta = Conv2D(bottle_dim, (1, 1),
                       padding='same',
                       kernel_initializer=initial()
                       )(input)
        theta = Reshape((-1, bottle_dim))(theta)
        phi = Conv2D(bottle_dim, (1, 1),
                     padding='same',
                     kernel_initializer=initial()
                     )(input)
        phi = Reshape((-1, bottle_dim))(phi)
        if compression > 1:
            phi = MaxPool1D(compression)(phi)
        f = Dot(axes=2)([theta, phi])
        f = Activation('softmax')(f)

    if mode == 'dot':
        theta = Conv2D(bottle_dim, (1, 1),
                       padding='same',
                       kernel_initializer=initial()
                       )(input)
        theta = Reshape((-1, bottle_dim))(theta)
        phi = Conv2D(bottle_dim, (1, 1),
                     padding='same',
                     kernel_initializer=initial()
                     )(input)
        phi = Reshape((-1, bottle_dim))(phi)
        f = Dot(axes=2)([theta, phi])
        size = K.int_shape(f)
        f = Lambda(lambda out: (1. / float(size[-1]) * out))(f)

    if mode == 'concatenate':
        theta = Conv2D(bottle_dim, (1, 1),
                       padding='same',
                       kernel_initializer=initial()
                       )(input)
        phi = Conv2D(bottle_dim, (1, 1),
                     padding='same',
                     kernel_initializer=initial()
                     )(input)
        new_x = Concatenate(axis=3)(theta, phi)
        f = Conv2D(bottle_dim, (1, 1),
                   padding='same',
                   kernel_initializer=initial()
                   )(new_x)
        f = Reshape((-1, bottle_dim))(f)
        f = Activation('relu')(f)

    # compute g
    g = Conv2D(bottle_dim, (1, 1),
               padding='same',
               kernel_initializer=initial()
               )(input)
    g = Reshape((-1, bottle_dim))(g)
    if compression > 1:
        g = MaxPool1D(compression)(g)

    # compute y
    y = Dot(axes=[2, 1])([f, g])
    y = Reshape((dim1, dim2, bottle_dim))(y)
    y = Conv2D(channel, (1, 1),
               padding='same',
               kernel_initializer=initial()
               )(y)

    # residual
    if residual == True:
        y = Add()([y, input])

    return y