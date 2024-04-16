import tensorflow as tf
from tensorflow.keras import layers, Model

class ExemplarColorizationModel:
    def create(self, input_shape, reference_shape):
        '''Creates and returns a model for exemplar-based video colorization.
        
        Args:
            input_shape: A tuple specifying the shape of the grayscale input frame.
            reference_shape: A tuple specifying the shape of the reference color frame.
        '''
        # Grayscale input
        grayscale_input = tf.keras.Input(shape=input_shape, name='grayscale_input')
        
        # Reference color input
        reference_input = tf.keras.Input(shape=reference_shape, name='reference_input')
        
        # Encoder for grayscale input
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(grayscale_input)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Encoder for reference input
        y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(reference_input)
        y = layers.MaxPooling2D((2, 2))(y)
        
        # Fusion of grayscale and reference features
        combined = layers.concatenate([x, y])
        
        # Decoder
        z = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(combined)
        z = layers.UpSampling2D((2, 2))(z)
        colorized_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='colorized_output')(z)
        
        # Model
        model = Model(inputs=[grayscale_input, reference_input], outputs=colorized_output, name='ExemplarColorizationModel')
        
        model.compile(optimizer='adam', loss='mse')
        
        return model