import importlib
from architecture.conv_autoencoder import ConvAutoencoder
from architecture.VGG16_transfer import VGG16_transfer


class Stack:
    '''This class holds the data, model, architecture, and results
    '''
    def __init__(self):
        #self.data = Data()
        
        self.architecture = None
        self.model = None
        
        self.final_model = None
        self.final_history = None
        
        self.train_list = None
        self.test_list = None
        self.val_list = None
        self.bw_train_list = None
        self.bw_test_list = None
        self.bw_val_list = None 
        
        self.img_width = None
        self.img_height = None
        
    def update_rgb(self, train_list, test_list, val_list):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.train_list = train_list
        self.test_list = test_list
        self.val_list = val_list
        
    def update_bw(self, bw_train_list, bw_test_list, bw_val_list):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.bw_train_list = bw_train_list
        self.bw_test_list = bw_test_list
        self.bw_val_list = bw_val_list    
    
    def create_model(self, hp, model_type='conv_autoencoder'):
        input_shape = (self.img_width, self.img_height, 1)
        if model_type is "conv_autoencoder":
            self.architecture = ConvAutoencoder(input_shape)
            #self.model = self.architecture.build_model(hp)
            self.model = self.architecture.build_model2()
        elif model_type is "VGG16_transfer":
            self.architecture = VGG16_transfer(input_shape)
            self.model = self.architecture.build_model(hp)
        elif model_type is "conv_autoencoder_simple":
            self.architecture = ConvAutoencoder(input_shape)
            self.model = self.architecture.build_model2(hp)
        
        return self.model
    
    def finished_model(self, final_model, final_history):
        '''Saves the final model and history for future use.'''
        self.final_model = final_model
        self.final_history = final_history
        
    def update_dimensions(self, width, height):
        self.img_width = width
        self.img_height = height

    def add_tuner(self, tuner):
        self.tuner = tuner
        