# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:17:35 2018

@author: Madhusudhan
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

def load_model(weights_path=None, shape=(128, 128)):
    # Part 1 - Building the CNN
        
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(64, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.5))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.5))
    
    # Adding a third convolutional layer (Only for cnnext weights)
    #classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    #classifier.add(MaxPooling2D(pool_size = (2, 2)))
    #classifier.add(Dropout(0.5))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units = 6, activation = 'softmax'))
    
    print ("Created model successfully")
    if weights_path:
        classifier.load_weights(weights_path)

    #Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return classifier

"""
    # Part 2 - Fitting the CNN to the images
    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_set = train_datagen.flow_from_directory('drive/fercnn1/dataset/train_set',
                                                  target_size = (128, 128),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory('drive/fercnn1/dataset/test_set',
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    
    classifier.fit_generator(train_set,
                             steps_per_epoch = 122,
                             epochs = 20,
                             validation_data = test_set,
                             validation_steps = 280)
    
    #classifier.save_weights('cnn_128x128_2convmax_2fc_64f_32bs_100e.h5')
    
    
    import numpy as np
    from keras.preprocessing import image
    
    im = image.load_img('test19.jpg', target_size = (128, 128))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis = 0)
    
    pr = classifier.predict(im)
    
"""