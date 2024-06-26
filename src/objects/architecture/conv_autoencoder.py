import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

class ConvAutoencoder:
    '''Will be the main class for our model'''
    def __init__(self):
        self.test = None
    
    def build_model(self, hp):
        print("HEREHREHERHEHRHEHRHEHRHEHRHEHRH")
        input_shape = (1080, 720, 1)  # Adjust based on your dataset
        inputs = Input(shape=input_shape)

        # Hyperparameters
        num_blocks = hp.Int('num_blocks', min_value=1, max_value=3, step=1)
        initial_num_filters = hp.Int('initial_num_filters', min_value=16, max_value=128, step=16)
        conv_layers = hp.Int('conv_layers', min_value=1, max_value=3, step=1)

        x = inputs

        # Encoder
        print(range(num_blocks))
        for i in range(num_blocks):
            num_filters = initial_num_filters * (2 ** i)
            # Hyperparameter
            filter_size = hp.Int('filter_size', min_value=3, max_value=5, step=1)
            
            for i in range(conv_layers):
                x = self.conv_block(x, num_filters, filter_size)
            x = MaxPooling2D((2, 2))(x)
            print("here")

        # Bottleneck
        num_filters *= 2
        for i in range(conv_layers):
            x = self.conv_block(x, num_filters, filter_size)

        # Decoder
        for i in reversed(range(num_blocks)):
            num_filters = initial_num_filters * (2 ** i)
            x = UpSampling2D((2, 2))(x)
            # Hyperparamter
            filter_size = hp.Int('filter_size', min_value=3, max_value=5, step=1)

            for i in range(conv_layers):
                x = self.conv_block(x, num_filters, filter_size)

        # Output layer
        outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                    loss="mse",  # Mean Squared Error is often used for reconstruction losses
                    metrics=['accuracy'])
        
        model.summary()

        return model
    
    def conv_block(self, input_tensor, num_filters, filter_size):
        x = Conv2D(num_filters, (filter_size, filter_size), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        return x
    
