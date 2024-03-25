import tensorflow as tf
from objects.result import Result
from preprocessing import preprocess

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.result = Result()

    def predict(self, input_data):
        '''Make predictions using the loaded model and input data.'''
        processed_data = preprocess(input_data)  # Use the imported function
        predictions = self.model.predict(processed_data)
        # Utilize Result class methods as needed
        self.result.save_predictions(predictions)  # Ensure this method exists in Result class
        return predictions
