import tensorflow as tf
import numpy as np
from Model import *


classes = ["COVID_19 +ve","COVID_19 -ve]
image_size = 224

x_train_path = input("Enter path to train images: ")
y_train_path = input("Enter path to train labels: ")
x_test_path = input("Enter path to test images: ")
y_test_path = input("Enter path to test labels: ")
path = input("Enter path to save the model: ")  #path to save the model


x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

inception = train_model(path, x_train, y_train,
                        x_test, y_test, model_name = "inception_v3",
                        epochs = 60, input_shape = (image_size,image_size,3),
                        classes = len(classes))

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, 
                           mode='min', restore_best_weights=True)
callbacks = [early_stop]

densenet = train_model(path, x_train, y_train,
                       x_test, y_test, model_name="densenet201",
                       epochs=60, input_shape = (image_size,image_size,3),
                       classes = len(classes),
                       callbacks = callbacks)

resnet = train_model(path, x_train, y_train,
                     x_test, y_test, model_name="resnet50_v2",
                     epochs=60, input_shape = (image_size,image_size,3),
                     classes = len(classes),
                     callbacks = callbacks)

