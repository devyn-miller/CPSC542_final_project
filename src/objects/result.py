import json
import matplotlib.pyplot as plt
import numpy as np

class Result:
    def __init__(self):
        self.predictions = None
        self.evaluation_metrics = None

    def save_predictions(self, predictions, file_path='predictions.json'):
        '''Saves the predictions made by the model to a JSON file.'''
        self.predictions = predictions
        with open(file_path, 'w') as f:
            json.dump(predictions.tolist(), f)  # Convert numpy array to list for JSON serialization

    def save_evaluation(self, evaluation_metrics, file_path='evaluation_metrics.json'):
        '''Saves the evaluation metrics obtained during validation to a JSON file.'''
        self.evaluation_metrics = evaluation_metrics
        with open(file_path, 'w') as f:
            json.dump(evaluation_metrics, f)

    def plot_predictions(self, sample_index=0):
        '''Plots the predictions for a given sample index.'''
        if self.predictions is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.predictions[sample_index])
            plt.title(f'Predictions for Sample {sample_index}')
            plt.ylabel('Prediction Value')
            plt.xlabel('Prediction Index')
            plt.show()
        else:
            print("No predictions to plot.")
