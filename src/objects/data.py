class Data:
    '''This class holds the data: train_dataset, test_dataset, val_dataset
    '''
    def __init__(self, images_color, images_b_w):
        self.images_color = images_color
        self.images_b_w = images_b_w
        
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        
        
    def update_datasets(self, train_dataset, test_dataset, val_dataset):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset