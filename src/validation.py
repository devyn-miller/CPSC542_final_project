import tensorflow as tf
from objects.result import Result
from preprocessing import preprocess 
class Validator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.result = Result()

    def validate(self, validation_data):
        '''Evaluates the model on the validation data.'''
        processed_data = preprocess(validation_data)  # Preprocess validation data
        evaluation_metrics = self.model.evaluate(processed_data)
        # Utilize Result class methods as needed, e.g., to save or visualize evaluation results
        self.result.save_evaluation(evaluation_metrics)  # Ensure this method exists in Result class
        return evaluation_metrics