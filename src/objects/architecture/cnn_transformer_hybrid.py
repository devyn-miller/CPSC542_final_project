import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from transformers import TransformerEncoder, TransformerEncoderLayer

class ConvTransformer:
    def __init__(self, input_shape, num_heads, d_model, num_encoder_layers, output_dim):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.output_dim = output_dim

    def create(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # CNN Part
        x = Rescaling(1./255)(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)

        # Transformer Part
        transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, num_heads=self.num_heads)
        transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=self.num_encoder_layers)
        x = transformer_encoder(x)

        # Output layer
        outputs = Dense(self.output_dim, activation='sigmoid')(x)

        model = Model(inputs, outputs, name='ConvTransformer')
        model.compile(optimizer='adam', loss='binary_crossentropy')

        return model