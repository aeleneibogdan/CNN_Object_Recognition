# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:13:51 2021

@author: aelen
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
import matplotlib.pyplot as plt

#Loading cifar10 dataset
(imageTrain, classTrain), (imageTest, classTest) = keras.datasets.cifar10.load_data()

print("Shape of training images is: ", imageTrain.shape) #(50000, 32, 32, 3)
print("Shape of testing images is: ", imageTest.shape)  #(10000, 32, 32, 3)

classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #10 classes

# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=0, baseline=0.9 )]

class TerminateOnBaselineCB(Callback):
    def on_epoch_end(self, epoch, logs= None ):
        logs = logs or {}
        acc = logs.get('accuracy')
        if acc is not None:
            if(acc > 0.9):
                print("Reached 90% accuracy. Training stops.")
                self.model.stop_training=True

#This function will be later used in plotting the image
def plot_sample(imageTrain, classTrain, index):
    plt.figure(figsize=(5,5))
    plt.imshow(imageTrain[index])
    plt.title(classes[yClass[index]])
    

#Normalizing the data into a 0 to 1 range
imageTrain = imageTrain / 255.0
imageTest = imageTest / 255.0

# Image augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


#Calling the callback
callbacks=TerminateOnBaselineCB()

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D(2,2),                           
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

#We use Adam optimizer as it provides better accuracy than Stochastic Gradient Descent optimizer

# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])

cnn = model.fit(imageTrain, classTrain, epochs=5, validation_data=(imageTest,classTest), callbacks=[callbacks])
model.evaluate(imageTest, classTest) # testing the accuracy on the test set

yTest = model.predict(imageTest) #predicting all the testing images and storing in yTest

yClass = [np.argmax(elem) for elem in yTest] #np.argmax returns the index of the maximum value

#printing yClass and classTest to compare the predicted results and the test data
print(yClass[:5])
print(classTest[:5])

#Change the index for another example
plot_sample(imageTest, classTest, 5) #plot the actual sample and the name of it
print("The predicted value is: ", classes[yClass[5]]) #the predicted value


#Summarize history for loss
plt.figure()
plt.title('Model Loss')
plt.plot(cnn.history['loss'])
plt.plot(cnn.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train','validation'], loc='upper left')
plt.show()

#Summarize history for accuracy
plt.figure()
plt.title('Model Accuracy')
plt.plot(cnn.history['accuracy'])
plt.plot(cnn.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','validation'], loc='upper left')
plt.show()
