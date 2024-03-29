from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras import *

def sat_class_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 4), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D((2, 2), padding = 'same'))
    
    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.1)) 
    model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    
    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.1))                  
    model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    
    model.add(Flatten()) 
    model.add(Dense(128, activation = 'linear'))
    model.add(LeakyReLU(alpha = 0.1))                  
    
    model.add(Dense(num_classes, input_shape = (3136, ), activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model
