import tensorflow as tf
from tensorflow.keras import layers, Model

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv3 = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')  # Output 3 channels for RGB

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return self.conv3(x)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(50, activation='relu')
        self.d2 = layers.Dense(1, activation='sigmoid')  # Binary classification

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(256, 256, 1))  # Example input shape
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# Instantiate the GAN components
generator = Generator()
discriminator = Discriminator()

# Create the GAN
gan = create_gan(generator, discriminator)