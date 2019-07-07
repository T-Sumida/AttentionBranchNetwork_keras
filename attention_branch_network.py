# coding:utf-8

from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import DenseNet121
from keras.applications import DenseNet169
from keras.applications import DenseNet201
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.models import Model


def create_attention_branch_net(arch_model, class_num):
    # Attention
    att_conv1 = Conv2D(512, (3, 3), padding="same", name='att_conv1')(arch_model.output)
    att_conv2 = Conv2D(512, (3, 3), padding="same", name='att_conv2')(att_conv1)
    att_bn1 = BatchNormalization(name='att_bn1')(att_conv2)
    att_conv3 = Conv2D(class_num, (1, 1), padding="same", name='att_conv3')(att_bn1)
    att_act1 = Activation('relu', name='att_act1')(att_conv3)

    att_conv4 = Conv2D(class_num, (1, 1), padding="same", name='att_conv4')(att_act1)
    att_gap = GlobalAveragePooling2D(name='att_gap')(att_conv4)
    att_output = Activation('softmax', name='att_output')(att_gap)

    att_conv5 = Conv2D(1, (1, 1), padding="same", name='att_conv5')(att_act1)
    att_bn2 = BatchNormalization(name='att_bn2')(att_conv5)
    att_map = Activation('sigmoid', name='att_map')(att_bn2)

    # Perception
    rx = Multiply(name='att_multiply')([arch_model.output, att_map])
    rx = Concatenate(name='att_concatenate')([arch_model.output, rx])
    perc_bn1 = BatchNormalization(name='perc_bn1')(rx)
    perc_act1 = Activation('relu', name='perc_act1')(perc_bn1)
    perc_avg1 = AveragePooling2D(name='perc_avg1')(perc_act1)
    perc_flatten = Flatten(name='perc_flatten')(perc_avg1)
    perc_fc = Dense(class_num, name='perc_fc')(perc_flatten)
    perc_output = Activation('softmax', name='perc_output')(perc_fc)

    model = Model(inputs=arch_model.input, outputs=[att_output, perc_output])

    print(model.summary())
    return model


def build_model(input_shape, output_num, feature_extractor='vgg16'):
    if feature_extractor == 'vgg16':
        arch = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    elif feature_extractor == 'vgg19':
        arch = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    elif feature_extractor == 'dense121':
        arch = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    elif feature_extractor == 'dense169':
        arch = DenseNet169(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    elif feature_extractor == 'dense201':
        arch = DenseNet201(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    return create_attention_branch_net(arch, output_num)

