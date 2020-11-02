from keras.layers import Input
from keras import layers
from keras.layers import Dense, Dropout, ZeroPadding2D
from keras.layers import Activation, add
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (1, 1), name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + "2a")(
        input_tensor
    )
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + "1")(
        input_tensor
    )
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("relu")(x)
    return x
