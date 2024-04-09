from data import Data
from objects.architecture.conv_autoencoder import ConvAutoencoder

class Stack:
    '''This class holds the data, model, architecture, and results
    '''
    def __init__(self):
        self.data = Data()
        
        self.architecture = None
        self.model = None
        
        self.final_model = None
        self.final_history = None
        
    def update_datasets(self, train_dataset, test_dataset, val_dataset):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        
    def create_model(self, hp, model_type='conv_autoencoder'):
        if model_type is "conv_autoencoder":
            self.architecture = ConvAutoencoder()
            self.model = self.architecture(hp)
            
            print("hello")
        elif model_type is "NoGAN":
            #call NoGAN.py for model creation
            print("hello")
        '''Creates a ML model using the given model type and updates self.model
        '''
        if model_type is 'ConvAutoencoder':
            '''Create the ConvAutoencoder architecture.'''

        return self.model
    
    def finished_model(self, final_model, final_history):
        '''Saves the final model and history for future use.'''
        self.final_model = final_model
        self.final_history = final_history
        

        