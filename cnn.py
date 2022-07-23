import torch
from torch import nn

def image_size(layer_data):


def build_cnn_layer(layer_data):
    layer = []

    layer.append(nn.Conv2d(layer_data.conv_params.input_channels,
                              out_channels=layer_data.conv_params.output_channels,
                              kernel_size = layer_data.conv_params.kernel_size,
                              stride = layer_data.conv_params.stride,
                              padding = layer_data.conv_params.padding)
    )

    if layer_data.activation_params is not None:
        if hasattr(layer_data.activation_params, "leaky_relu") and layer_data.activation_params.leaky_relu is True:
            layer.append(nn.LeakyReLU(inplace=True))
        elif hasattr(layer_data.activation_params, "relu") and layer_data.activation_params.relu is True:
            layer.append(nn.ReLU(inplace=True))

    if layer_data.pooling_params is not None:
        layer.append(nn.MaxPool2d(kernel_size=layer_data.pooling_params.kernel_size,
                                    stride=layer_data.pooling_params.stride,
                                    padding=layer_data.pooling_params.padding)
        )

    if use_batching is True:
        layer.append(nn.BatchNorm2d(layer_data.conv_params.output_channels))

    return nn.Sequential(*layer)
