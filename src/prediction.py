import tensorflow as tf
from src.objects.result import Result
from src.preprocess.preprocessing import preprocess

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.result = Result()

    def predict(self, input_data):
        '''Make predictions using the loaded model and input data.'''
        processed_data = preprocess(input_data)  # Use the imported function
        predictions = self.model.predict(processed_data)
        # Utilize Result class methods as needed
        if hasattr(self.result, 'save_predictions'):
            self.result.save_predictions(predictions)
        else:
            raise AttributeError("Result class does not have a 'save_predictions' method")
        return predictions
