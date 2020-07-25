import numpy as np 
import tensorflow as tf 

image_size = 224
classes = ["COVID_19 +ve","COVID_19 -ve"]

def ensemble(x, weights, models): 
    '''
    returns a weighted average of predictions made by the models\n
    x -> input image \n
    weights -> a list of weights \n
    models -> a list of models\n    
    '''      
    outputs = []    
    for model in models:                
        outputs.append(list(model.predict(x)[0]))                
    
    outputs = np.array(outputs)
    avg = np.average(a=outputs,axis=0,weights=weights)
    return avg


def equal(pred, label):
  pred_id = np.argmax(pred)
  if (pred_id == label):
    return True
  else:
    return False


def accuracy(predicted_values, y_truths):
  '''
  returns accuracy\n
  predicted_values = a numpy array containing the predictions\n
  y_truths = a numpy array containing the truth values\n  
  '''
  total = len(y_truths)
  correct = 0
  for i in range(len(y_truths)):
    if equal(predicted_values[i],y_truths[i]):
      correct += 1
  acc = correct / total
  return acc


def generate_weights(x_val, y_val, models):
    '''
    returns a list of weights
    '''
    accuracy = []
    weights = np.full((1,len(models)), 100.0)
    for model in models:
      acc = model.evaluate(x_val, y_val)[1]
      accuracy.append(100*acc)
    weights = weights - accuracy
    weights = weights**2
    sum = np.sum(weights)
    weights = weights/sum
    weights = 1/weights
    weights = weights**2
    sum = np.sum(weights)
    weights = weights/sum
    return weights

