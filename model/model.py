import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Softmax, BatchNormalization, ReLU
from layers.depthwise_conv_block import DepthWiseConvBlock
import pdb


class MobileNetV1(Model):
    def __init__(self, input_shape, alpha=1.25, depth_multiplier=1, include_top=True, classes=37):
        super(Model, self).__init__()
        self.include_top = include_top

        self.input_layer = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=input_shape)
        self.batch_norm_layer = BatchNormalization()
        self.relu_layer = ReLU()
        self.separable_conv_layer1 = DepthWiseConvBlock(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer2 = DepthWiseConvBlock(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer3 = DepthWiseConvBlock(128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer4 = DepthWiseConvBlock(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer5 = DepthWiseConvBlock(256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer6 = DepthWiseConvBlock(256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer7 = DepthWiseConvBlock(512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer8 = DepthWiseConvBlock(512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer9 = DepthWiseConvBlock(512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                        width_multiplier=alpha)
        self.separable_conv_layer10 = DepthWiseConvBlock(512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                         width_multiplier=alpha)
        self.separable_conv_layer11 = DepthWiseConvBlock(512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                         width_multiplier=alpha)
        self.separable_conv_layer12 = DepthWiseConvBlock(1024, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                         width_multiplier=alpha)
        self.separable_conv_layer13 = DepthWiseConvBlock(1024, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                         width_multiplier=alpha)
        if self.include_top:
            self.pool_layer = AveragePooling2D((7, 7))
            self.dense = Dense(classes)
            self.softmax = Softmax()

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.batch_norm_layer(x)
        x = self.relu_layer(x)
        x = self.separable_conv_layer1(x)
        x = self.separable_conv_layer2(x)
        x = self.separable_conv_layer3(x)
        x = self.separable_conv_layer4(x)
        x = self.separable_conv_layer5(x)
        x = self.separable_conv_layer6(x)
        x = self.separable_conv_layer7(x)
        x = self.separable_conv_layer8(x)
        x = self.separable_conv_layer9(x)
        x = self.separable_conv_layer10(x)
        x = self.separable_conv_layer11(x)
        x = self.separable_conv_layer12(x)
        x = self.separable_conv_layer13(x)
        if self.include_top:
            x = self.pool_layer(x)
            x = self.dense(x)
            x = self.softmax(x)

        return x
