def image_size_after_convolution(image_size, kernel_size, padding, stride):
    return int((image_size - kernel_size + (padding * 2)) / stride) + 1

def image_sizes_after_convolutions(image_size, kernel_sizes, paddings, strides):
    current_size = image_size
    image_sizes = []
    for i in range(len(kernel_sizes)):
        current_size = image_size_after_convolution(current_size, kernel_sizes[i], paddings[i], strides[i])
        image_sizes.append(current_size)
    return image_sizes
