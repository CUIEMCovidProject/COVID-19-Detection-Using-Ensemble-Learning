import tensorflow as tf 
from tensorflow.keras.applications import VGG19, ResNet50V2, DenseNet201, InceptionResNetV2, InceptionV3, Xception, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

def train_model(path, train_images=None, train_labels = None, 
                test_images = None, test_labels = None, 
                model_name = None, epochs =80, learning_rate = 0.0001,
                input_shape = (224,224,3), classes=2, batch_size = 16, 
                classifier_activation='softmax',
                callbacks = None):
    '''    
    saves the model as .h5 file\n  
    path = directory for saving the files
    train_images = a numpy array containing the image data for training\n
    train_labels = a numpy array containing the labels for training\n
    test_images = a numpy array containing the image data for test\n
    test_labels = a numpy array containing the labels for test\n
    model_name = a string, name of the model -> "vgg19", "resnet50_v2", "inception_resnet_v2", "densenet201", "inception_v3", "xception", "mobilenet_v2"\n
    epochs\n
    learning_rate\n        
    '''

    base_model = None
    if model_name == 'vgg19':        
        base_model = VGG19(weights = None, include_top = False, input_shape = input_shape)
     
    if model_name == 'resnet50_v2':
        base_model = ResNet50V2(weights = None, include_top = False, input_shape = input_shape)
             
    if model_name == 'inception_resnet_v2':        
        base_model = InceptionResNetV2(weights = None, include_top = False, input_shape = input_shape)
    
    if model_name == 'densenet201':        
        base_model = DenseNet201(weights = None, include_top = False, input_shape = input_shape)
          
    if model_name == 'inception_v3':        
        base_model = InceptionV3(weights = None, include_top = False, input_shape = input_shape)
        
    if model_name == 'xception':        
        base_model = Xception(weights = None, include_top = False, input_shape = input_shape)
        
    if model_name == 'mobilenet_v2':        
        base_model = MobileNetV2(weights = None, include_top = False, input_shape = input_shape)

    x = base_model.output         
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(classes, activation=classifier_activation)(x)

    model = tf.keras.Model(inputs = base_model.input, outputs = output)

    optimizer = Adam(learning_rate = learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(optimizer = optimizer,
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'])
        
    results = model.fit(train_images, train_labels, epochs = epochs,
                        validation_data = (test_images, test_labels), 
                        batch_size=batch_size, 
                        callbacks = callbacks
                        )
    
    #losses = pd.DataFrame(model.history.history)
    #losses[['loss','val_loss']].plot()
    
    save_model = path + model_name + '.h5'
    model.save(save_model)
    
    return results    