import tensorflow as tf
import importlib

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import objects.stack as stack
importlib.reload(stack)
from objects.stack import Stack

import shutil
import os

HAYDENS_COMPUTER = False

if HAYDENS_COMPUTER:
    import os
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
def delete_unet_tuning_folder():
    folder_path = './src/models/unet_tuning'
    
    # Check if the folder exists
    if os.path.exists(folder_path):
        try:
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print("Folder 'unet_tuning' has been deleted successfully.")
        except Exception as e:
            print(f"An error occurred while trying to delete the folder: {e}")
    else:
        print("The folder 'unet_tuning' does not exist.")
    
def black_and_white_generator(color_generator, batch_size=8):
    """Yield pairs of grayscale and original color images from an already normalized color generator."""
    for image in color_generator:
        grayscale_image = tf.image.rgb_to_grayscale(image)

        yield (np.array(grayscale_image), np.array(image))

def generator_to_list(generator):
    """
    Converts a generator yielding pairs of grayscale and original color images
    into two separate lists: one for grayscale and one for color images.
    
    Args:
        generator (generator): A generator object yielding tuples of (grayscale_image, color_image).

    Returns:
        tuple: A tuple containing two lists, (list of grayscale images, list of color images).
    """
    grayscale_images = []
    color_images = []
    for grayscale_image, color_image in generator:
        grayscale_images.append(np.array(grayscale_image))
        color_images.append(np.array(color_image))
    
    return grayscale_images, color_images

def run_tuner(stack, c):
    '''This is a turner.  It allows you to train up multiple models so 
    that you can figure out an architecture that works for you.
    '''
    #delete_unet_tuning_folder()
    
    tuner = kt.RandomSearch(
        stack.create_model,
        objective='val_accuracy',
        max_trials=c["max_trials"],  # Adjust as necessary
        executions_per_trial=c["executions_per_trial"],  # Adjust as necessary for reliability
        directory='./src/models',
        project_name='unet_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=c["patience"])

    bw_train_gen = black_and_white_generator(stack.train_generator)
    bw_val_gen = black_and_white_generator(stack.val_generator)

    grays_test_list, color_test_list = generator_to_list(bw_train_gen)
    gray_val_list, color_val_list = generator_to_list(bw_val_gen)

    train_inputs, train_targets = color_test_list, grays_test_list
    val_inputs, val_targets = color_val_list, gray_val_list

    # Call the tuner's search function
    tuner.search(
        x=train_inputs,  # Inputs for the model (Color images)
        y=train_targets,  # Targets for the model (Grayscale images)
        epochs=c["epochs"],  # Adjust epochs according to your configuration needs
        validation_data=(val_inputs, val_targets),  # Validation data as a tuple of (inputs, targets)
        callbacks=[stop_early]  # List of callbacks, including early stopping
    )

    return tuner

def get_best_model(stack):
    c = {
        "max_trials": 20,
        "executions_per_trial": 2,
        "epochs": 50,
        "patience": 5,
        "m": 5
    }
    '''c = {
        "max_trials": 4,
        "executions_per_trial": 1,
        "epochs": 2,
        "patience": 1,
        "m": 1
    }'''
    
    tuner = run_tuner(stack, c)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = stack.create_model(best_hps)
    
    stop_early = EarlyStopping(monitor='val_loss', patience=c["patience"]*c["m"])

    history = best_model.fit(
        black_and_white_generator(stack.train_generator), stack.train_generator,
        steps_per_epoch=stack.train_generator.samples // stack.train_generator.batch_size,
        epochs=c["epochs"]*c["m"],  # Train for more epochs
        validation_data=(black_and_white_generator(stack.val_generator), stack.val_generator),
        validation_steps=stack.val_generator.samples // stack.val_generator.batch_size,
        callbacks=[stop_early]
    )

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model.save(f'../models/model_{datetime_str}.weights.h5')
    stack.finished_model(best_model, history)
    return stack

def evaluate_model_performance(model, dataset_val):
    val_loss, val_accuracy = model.evaluate(dataset_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

def plot_training_history(history, metric='accuracy', val_metric='val_accuracy', 
                          loss='loss', val_loss='val_loss', title_suffix=''):
    # Plot specified metric values
    plt.figure(figsize=(12, 5))
    
    # Plot for the provided metric
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric])
    plt.plot(history.history[val_metric])
    plt.title(f'Model {metric.capitalize()} {title_suffix}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot for the loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history[loss])
    plt.plot(history.history[val_loss])
    plt.title(f'Model Loss {title_suffix}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()