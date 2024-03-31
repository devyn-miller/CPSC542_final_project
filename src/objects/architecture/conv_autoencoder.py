import tensorflow as tf
from tensorflow.keras import layers, Model

class ConvAutoencoder:
    '''Defines the architecture of a convolutional autoencoder model.'''

    def create(self, hp):
        '''Creates and returns a convolutional autoencoder model.
        
        Args:
            hp: A dictionary or an object containing hyperparameters.
        '''
        # Encoder
        inputs = tf.keras.Input(shape=(hp['input_shape']))
        x = layers.Conv2D(filters=hp['encoder_filters'], kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2D(filters=hp['decoder_filters'], kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        outputs = layers.Conv2D(filters=hp['output_channels'], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        
        # Model
        model = Model(inputs, outputs, name='ConvAutoencoder')
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        return model