import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def augment(X_train): 
    datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    )

    datagen.fit(X_train)
    
    # X_train_sub = X_train[:10]

    # plt.figure(figsize=(20,2))
    # for i in range(10):
    #     plt.subplot(1, 10, i+1)
    #     plt.imshow(X_train_sub[i])
    #     plt.suptitle('Original Training Images', fontsize=15)
    #     plt.show()

    # # Augmented Data
    # plt.figure(figsize=(20,2))
    # for X_batch in datagen.flow(X_train_sub, batch_size=12):
    #     for i in range(10):
    #         plt.subplot(1, 10, i+1)
    #         plt.imshow(X_batch[i])
    #     plt.suptitle('Augmented Images', fontsize=15)
    #     plt.show()
    #     break    
    
    # return