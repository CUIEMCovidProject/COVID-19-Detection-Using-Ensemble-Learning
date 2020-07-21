import tensorflow as tf 
import numpy as np

categories = ["COVID +ve","COVID -ve"]
image_size = 224

inception_path = './SavedModels/inception_v3.h5'
resnet_path = './SavedModels/resnet50_v2.h5'
densenet_path = './SavedModels/densenet201.h5'
print("Loading Models.....")
inception_model = tf.keras.models.load_model(inception_path)
resnet_model = tf.keras.models.load_model(resnet_path)
densenet_model = tf.keras.models.load_model(densenet_path)
models = [inception_model,resnet_model,densenet_model]
models_name = ["Inception","Resnet","DenseNet"]
print("Models Loaded")

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

def predict(input_image):
    '''
    returns a predicted class, associated probablity 
    input_image = an image
    models = list of models    
    '''
    weights = [0.172,0.601,0.228]
    input_image = input_image/255.0  
    img = input_image.reshape(-1,image_size,image_size,3)    
    model_predictions = []
    model_probs = []
    for i in range(len(models)):
        print(models_name[i], categories[np.argmax(models[i].predict(img))])
        model_predictions.append(categories[np.argmax(models[i].predict(img))])
        model_probs.append(models[i].predict(img))
    avg_pred = ensemble(img,weights,models)
    if model_predictions.count(categories[0]) == 1:        
        print(categories[0], model_probs[model_predictions.index(categories[0])][0])
    return categories[np.argmax(avg_pred)], avg_pred



