from src.VGG16TransferColorizer.VGG16TransferColorizerModel import VGG16TransferColorizer
from VGG16TransferColorizer_main import *
from src.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, gray2rgb

IMAGE_SHAPE: tuple = (360,640)
MODEL_FILE: str = 'vgg16_colorizer_model.keras'
old_colorizer = VGG16TransferColorizer(image_shape=IMAGE_SHAPE)
old_colorizer.load_model(MODEL_FILE)

colorizer = VGG16TransferColorizer(image_shape=IMAGE_SHAPE)
colorizer.build_model()
print(colorizer.model.output)
# Load the pre-trained weights into the modified model
colorizer.model.set_weights(old_colorizer.model.get_weights())
colorizer.is_trained = True

'''
VALIDATION
'''

from PIL import Image
import numpy as np

def load_image(image_path, image_shape):
    """
    Loads an image from the specified file path and resizes it to the given shape.

    Args:
        image_path (str): The file path of the image to be loaded.
        image_shape (tuple): The desired shape of the loaded image (height, width).

    Returns:
        numpy.ndarray: The loaded and resized image as a NumPy array.
    """
    # Open the image using PIL
    image = Image.open(image_path)

    # Resize the image to the desired shape
    image = image.resize(image_shape)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize the pixel values to the range [0, 1]
    image_array = image_array.astype('float32') / 255.0

    return image_array


# Make predictions
# Continuously predict and visualize new images

data_dir = 'vibrant'
image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

fig, ax = plt.subplots(figsize=(10, 6))

for image_path in image_paths:
    test_image = load_image(image_path, IMAGE_SHAPE)
    predicted_color = colorizer.predict(test_image)

    # Display the predicted color image
    ax.clear()
    ax.imshow(predicted_color)
    ax.set_title('Predicted Color Image')
    ax.axis('off')
    plt.pause(2)  # Pause for 2 seconds before updating the image
