from tensorflow.keras.layers import Layer, DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, ZeroPadding2D


class DepthWiseConvBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', width_multiplier=1.0):
        super(DepthWiseConvBlock, self).__init__()
        self.filters = int(filters * width_multiplier)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.depthwise_conv_layer = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides,
                                           padding=self.padding)
        self.batch_norm1_layer = BatchNormalization()
        self.relu_layer = ReLU()
        self.pointwise_conv_layer = Conv2D(self.filters, kernel_size=(1, 1), strides=(1, 1))
        self.batch_norm2_layer = BatchNormalization()
        self.relu2_layer = ReLU()

    def call(self, inputs):
        x = self.depthwise_conv_layer(inputs)
        x = self.batch_norm1_layer(x)
        x = self.relu_layer(x)
        x = self.pointwise_conv_layer(x)
        x = self.batch_norm2_layer(x)

        return self.relu2_layer(x)