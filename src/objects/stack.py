from src.objects.data import Data
from src.objects.architecture.conv_autoencoder import ConvAutoencoder

class Stack:
    '''This class holds the data, model, architecture, and results
    '''
    def __init__(self):
        self.data = Data()
        
        self.architecture = None
        self.model = None
        
        self.final_model = None
        self.final_history = None
        
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        
    def update_datasets(self, train_generator, test_generator, val_generator):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.val_generator = val_generator
        
    def create_model(self, hp, model_type='conv_autoencoder'):
        if model_type is "conv_autoencoder":
            self.architecture = ConvAutoencoder()
            self.model = self.architecture(hp)
        
        return self.model
    
    def finished_model(self, final_model, final_history):
        '''Saves the final model and history for future use.'''
        self.final_model = final_model
        self.final_history = final_history
        

        