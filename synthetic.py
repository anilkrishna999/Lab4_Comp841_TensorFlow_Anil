# File Name: Synthetic.py
# Author: Anil Krishna
# Date: 9th Nov 2021

"""
Simple Linear Regression with Synthetic Data
"""

# Import relevant modules
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


# Define the functions that build and train a model
def build_model(my_learning_rate):
    """
    Create and compile a simple linear regression model.
    """
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epoch, batch_size):
    """
    Train the model by feeding it data.
    """

    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epoch)

    # Gather the trained model's weight and bias.
    train_weight = model.get_weights()[0]
    train_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epoch = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return train_weight, train_bias, epoch, rmse


# Define the plotting functions
def plot_the_model(train_weight, train_bias, feature, label):
    """Plot the trained model against the training feature and label."""

    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    feature0 = 0
    label0 = train_bias
    feature1 = feature[-1]
    label1 = train_bias + (train_weight * feature1)
    plt.plot([feature0, feature1], [label0, label1], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epoch, rms_eror):
    """
    Plot the loss curve, which shows loss vs. epoch.
    """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epoch, rms_eror, label="Loss")
    plt.legend()
    plt.ylim([rms_eror.min() * 0.97, rms_eror.max()])
    plt.show()


my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
               12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8,
             38.2])

LEARNING_RATE = 0.01
EPOCHS = 500
MY_BATCH_SIZE = 12

my_model = build_model(LEARNING_RATE)
trained_weight, trained_bias, epochs, rms_error = train_model(my_model,
                                                              my_feature,
                                                              my_label,
                                                              EPOCHS,
                                                              MY_BATCH_SIZE)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rms_error)
