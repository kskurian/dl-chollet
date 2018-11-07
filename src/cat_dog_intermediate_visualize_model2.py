from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())

## Now we need to take an image of cat that wasnt used to train the model

img_path = 'C:/Users/10126/PycharmProjects/all/test1/45.jpg'

from keras_preprocessing import image
import numpy as np

img = image.load_img(img_path,target_size=(150,150))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

print(img_tensor.shape)

import matplotlib.pyplot as plt

#plt.imshow(img_tensor[0])
#plt.show()

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, output=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
#plt.show()
#plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
#plt.show()

## Here we get the layer names
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# The feature map has shape (1, size, size, n_features).

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
