# Identifying Drive-Thru Buildings with Satellite Imagery Recognition Model
## Introduction
Commercial real estate (CRE) professionals have to analyze thousands of potential retail sites to know which one has the highest financial potential. It is crucial for them to quickly understand every possible building characteristic that may affect the site's potential. There are hundreds of building features they may be interested in, this article will demonstrate how you can use satellite imagery paired with an image recognition model to find if a building has a Drive-Thru.

Using the power of AI image recognition real estate professionals can save hundreds of hours compiling data about their and potential properties. They can use the data this algorithm produces to speed up crucial decisions, get insights into market conditions, and understand the true financial potential of a site.

Other image recognition algorithms we found can only identify that our images are satellite images. They are not specific enough to recognize building features in the image. We build this specialized algorithm to recognize what is in the satellite image and in this particular example if we can find a drive-thru at this location.

https://miro.medium.com/v2/resize:fit:720/format:webp/1*253HvhuiVRZrJaV9_6zryg.jpeg
Fast food (QSR) restaurant with a drive-thru
## About the Algorithm
The objective of this study is to create an algorithm to substitute human data collection of building features with artificial intelligence satellite image recognition. This convolutional neural network (CNNs) drive-thru predicted with an accuracy of 72% whether a building had a drive-thru or not from only knowing the address of the building.

Data Preparation and Description of Variables
Building data were collected from web scrapers. Satellite images were pulled using Google API. We tested the performance of three different classification model architectures before creating the final model to ensure we had the optimal model for the task. Y

To avoid selection bias, a data sample from a larger dataset for each model was selected randomly. The data were up-sampled to create an even distribution of each category so that the algorithm had enough examples of each class to learn from.

We converted addresses to geocodes to get latitude and longitude coordinates for each address. Those coordinates were used as inputs into Google API to download map and satellite images of size 200x200 for each building in the model samples. We tested the models on images of different sizes. Size 200x200 had enough details for the models to train on and increasing the image size did not improve the performance of the models. The images were then converted to black and white to decrease storage size and training processing requirements which allowed us to use more images to train the model. You can read more about how the data was prepared here.

We also utilized a data augmentation algorithm, which rotates each image three times to increase the number of examples that the models get trained on.

## Model Architectures
Classification models were built using a convolutional neural network (CNN) technique. We tested 3 different neural network architectures to see if deeper models with more parameters would improve the prediction. Based on the comparison of the architecture of the models, we created the final model.

The first shallow CNN model we tested was a modification of the handwritten digits classification model based on Modified National Institute of Standards and Technology (MNIST) dataset described by Yalçın (2018). Table A1 describes the architecture of our shallow model. We added several convolutional layers because the patterns of our images are more complicated than the digits that the NMIST model was predicting. The model picked up some of the image patterns but the accuracy was not high and thus, not very useful.

To improve the accuracy of predictions of the first shallow model, we tested a deeper CNN model architecture from Brownlee (2019). Table B describes the architecture of this model.

The Final drive-thru Model
The drive-thru model allows the user to classify whether a building has a drive-thru with a prediction accuracy of 72% on the holdout set. First, you will need to import some dependencies and the data that was prepared in the previous blog article.

import requests
import pandas as pd

import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
%matplotlib inline
from skimage import data, io, filters, color, exposure


from numpy import array
import pickle  

import keras
##from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
##import tensorflow as tf
pkl_file = open('sb_buildings_drivein_geocoded_img.pkl', 'rb')  
all_images= pickle.load(pkl_file)
pkl_file.close()  

pkl_file = open('sb_buildings_drivein_geocoded_tag.pkl', 'rb') 
file_names= pickle.load(pkl_file)
pkl_file.close()  
plt.imshow(all_images[6])
The drive-thru model was trained on 3,637 images. Data were up-sampledThe drive-thru model was trained on 3,637 images. Data were up-sampled so that both classes of the dependent variable had a relatively equal count. The train and test ratio for this model was 70:30. Data were augmented with three rotations of each image to increase the training size and improve the performance of the model.

Image Prep

all_image_gray=[]
for image in all_images:
    all_image_gray.append(color.rgb2gray(image))
all_images_gray_array=array(all_image_gray)   
Tag Prep

file_names_array=array(file_names) 
pd.value_counts(pd.Series(file_names_array))
Sample prep

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_images_gray_array, file_names_array, test_size=0.3)


img_rows, img_cols = 250, 250
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train= encoder.fit_transform(y_train)
y_train=np.hstack((y_train, 1 - y_train))
y_test= encoder.fit_transform(y_test)
y_test=np.hstack((y_test, 1 - y_test))
print(y_train)
Augmenting the data

x_train_rot=[]
y_train_rot=[]
for i in range(len(x_train)):
    x_train_rot.append(x_train[i])
    y_train_rot.append(y_train[i])
               
    x_train_rot.append(x_train[i][::-1,:,:])
    y_train_rot.append(y_train[i])
    
    x_train_rot.append(x_train[i][:,::-1,:])
    y_train_rot.append(y_train[i])
               
    x_train_rot.append(x_train[i].transpose([1,0,2]))
    y_train_rot.append(y_train[i])
x_train_rot = np.array(x_train_rot,dtype='float32')
y_train_rot = np.array(y_train_rot,dtype='float32')
len(x_train_rot)


The model was built using both shallow and deep CNN architectures. The performance of the model using these two architectures wasn’t satisfactory due to very low prediction accuracy and overfitting. The final model that improved accuracy and reduced overfitting was built using an architecture that comprises four convolutions, two drop out, two max pool, and two dense layers. The model also utilized early stopping to tackle the problem of overfitting. Table F describes the architecture of the final model.


## Model Architecture

batch_size =20

num_classes = 2

epochs = 30


model = Sequential()
model.add(Conv2D(32, kernel_size=(10, 10),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(10, 10),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(10, 10)))
model.add(Dropout(0.25))


model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train_rot, y_train_rot,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

We created specialized code to understand the quality of the model

### Model Evaluation
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
### Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
predicted_classes = model.predict_classes(x_test)
y_test_list=[]
for i in range(len(y_test)):
    if ((y_test[i][0])==1):
        y_test_list.append(1)
    if ((y_test[i][1])==1):
        y_test_list.append(2)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test_list,predicted_classes))
model.save('0032 capstone_drivethru_classifier.h5')
The classification confusion matrix in Table G describes in more detail the performance of the model for each class. Chart 2 below shows the increase in training and validation accuracy of the model with each epoch.

### Table F: Drive-Thru Final Model Architecture
https://miro.medium.com/v2/resize:fit:640/format:webp/1*IbhXIBqZsewXk7z212N72w.jpeg

## Conclusions
We can identify building features from satellite images with reasonable accuracy. These deep learning techniques can replace human site surveys to classify, measure and identify building features for commercial real estate decision-making and analytics. Limitations of geocoding and outdated images bring a fair amount of noise to the data that is outside of the scope of the models. Showing the models more examples will help improve the accuracy and help them generalize.

More advanced specialized image recognition algorithms often reach over 95% accuracy so our prediction accuracy of 72% falls short of that mark. Yet, these models can provide actionable insights to our industry partners. Considering the variety of buildings across the US and some of the limitations of geocodes and image vintages mentioned before, the models outperformed our expectations. The results of the models we built are almost as good as some of the more generic algorithms out there, in identifying these specific building features.

These models can substitute in-person site visits to identify building features at the preliminary stages of real estate selection thus saving companies time and resources. Real estate decision-makers can use the results of this algorithm to quickly filter and find desired buildings. Identifying these building features can be used as inputs to improve.

References.

Brownlee, J. (2019, May 20). How to Develop a Deep CNN for Multi-Label Classification of Photos [Blog post].

Google. Maps Static API.

Yalçın, O. (2018, August 19). Image Classification in 10 Minutes with MNIST Dataset [Blog post].

Original Post https://miro.medium.com/v2/resize:fit:640/format:webp/1*IbhXIBqZsewXk7z212N72w.jpeg
