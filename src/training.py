import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

from datetime import datetime
import matplotlib.pyplot as plt

from objects.stack import Stack

HAYDENS_COMPUTER = True

if HAYDENS_COMPUTER:
    import os
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





def run_tuner(stack, c):
    '''This is a turner.  It allows you to train up multiple models so 
    that you can figure out an architecture that works for you.
    '''
    tuner = kt.RandomSearch(
        stack.create_model,
        objective='val_accuracy',
        max_trials=c["max_trials"],  # Adjust as necessary
        executions_per_trial=c["executions_per_trial"],  # Adjust as necessary for reliability
        directory='../models',
        project_name='unet_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=c["patience"])

    tuner.search(
        stack.train_generator,
        steps_per_epoch=stack.train_generator.samples // stack.train_generator.batch_size,
        epochs=c["epochs"],  # Adjust epochs according to your need
        validation_data=stack.val_generator,
        validation_steps=stack.val_generator.samples // stack.val_generator.batch_size,
        callbacks=[stop_early]
    )

    return tuner

def get_best_model(stack):
    '''c = {
        "max_trials": 4,
        "executions_per_trial": 1,
        "epochs": 10,
        "patience": 3
        "m": 5
    }'''
    c = {
        "max_trials": 4,
        "executions_per_trial": 1,
        "epochs": 2,
        "patience": 1,
        "m": 1
    }
    
    tuner = run_tuner(stack, c)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = stack.create_model(best_hps)
    
    stop_early = EarlyStopping(monitor='val_loss', patience=c["patience"]*c["m"])

    history = best_model.fit(
        stack.train_generator,
        steps_per_epoch=stack.train_generator.samples // stack.train_generator.batch_size,
        epochs=c["epochs"]*c["m"],  # Train for more epochs
        validation_data=stack.val_generator,
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