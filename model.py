import os
import csv
from scipy import ndimage
import numpy as np
import sklearn
import random

samples = []
with open('../auto_clone_data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

## 20% of the recoded images as validation and 80% of the recoded images as training
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../auto_clone_data2/IMG/'+batch_sample[0].split('/')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

#creat model
model = Sequential()
#normalized and mean_centered
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
#crop each image 
model.add(Cropping2D(cropping=((70,25),(0,0))))  
#Convolution
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))       
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) 
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu")) 
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu")) 
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu")) 

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))          

#use MSE as loss function            
model.compile(loss='mse',optimizer='adam')

#training
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
