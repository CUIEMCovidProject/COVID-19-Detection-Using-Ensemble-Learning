import tensorflow as tf 
import numpy as np
from Ensembling import *
from sklearn.metrics import classification_report, confusion_matrix

x_test_path = input("Enter path to test images: ")
y_test_path = input("Enter path to test labels: ")

inception_path = input("Enter path to Inception Model: ")
resnet_path = input("Enter path to Resnet Model: ")
densenet_path = input("Enter path to DenseNet Model: ")

x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

image_size = 224

inception_model = tf.keras.models.load_model(inception_path)
resnet_model = tf.keras.models.load_model(resnet_path)
densenet_model = tf.keras.models.load_model(densenet_path)

models = [densenet_model,resnet_model,inception_model]

w = gen_weights(x_test,y_test,models) #generating weights
print("Weights: ", w)

predictions = []
for i in range(len(x_test)):
  pred = ensemble(x_test[i].reshape(-1,image_size,image_size,3),w,models)
  predictions.append(pred)

print("Accuracy: ",accuracy(predictions,y_test))

y_pred = np.argmax(np.array(predictions), axis=1)

print("The classification report: ")
print(classification_report(y_pred=y_pred, y_true=y_test))
print()
print("Confusion Matrix: ")
print(confusion_matrix(y_pred=y_pred, y_true=y_test))