import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D ,MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle
import pandas as pd
import numpy as np
import random

# cloning repo containing the traffic sign dataset
!git clone https://bitbucket.org/jadslim/german-traffic-signs

# list dataset contents
 
!ls german-traffic-signs

#load the dataset

data = pd.read_csv('german-traffic-signs/signnames.csv')

data

#loading pickled dataset

with open("german-traffic-signs/train.p",mode='rb') as training:
  train = pickle.load(training)

with open("german-traffic-signs/valid.p",mode='rb') as validation:
  valid = pickle.load(validation)

with open("german-traffic-signs/test.p",mode='rb') as testing:
  test = pickle.load(testing)

X_train , y_train = train['features'], train['labels']
X_validation , y_validation = valid['features'], valid['labels']
X_test , y_test = test['features'], test['labels']

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)

index = np.random.randint(1,len(X_train))
plt.imshow(X_train[index])
print("image label={}".format(y_train[index]))

# shuffle the data
from sklearn.utils import shuffle
X_train , y_train = shuffle(X_train,y_train)

def preprocessing(img): 
  #convert to grayscale
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #Applying histogram equalization 
  img = cv2.equalizeHist(img)
  #Normalization
  img = img/255
  return img

X_train_processed = np.array(list(map(preprocessing,X_train)))
X_validation_processed = np.array(list(map(preprocessing,X_validation)))
X_test_processed = np.array(list(map(preprocessing,X_test)))

X_train_processed = X_train_processed.reshape(34799,32,32,1)
X_test_processed = X_test_processed.reshape(12630,32,32,1)
X_validation_processed = X_validation_processed.reshape(4410,32,32,1)

print(X_train_processed.shape)
print(X_test_processed.shape)
print(X_validation_processed.shape)

i= random.randint(1,len(X_train))
plt.imshow(X_train_processed[i].squeeze(),cmap='gray')
plt.figure()
plt.imshow(X_train[i].squeeze())

model = Sequential()
# add the convolutional layer
# filters, size of filters,input_shape,activation_function
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,1)))

# pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# place a dropout layer
model.add(Dropout(0.25))

# add another convolutional layer
model.add(Conv2D(64,(5,5),activation='relu'))

# pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the image to 1 dimensional array
model.add(Flatten())

# add a dense layer : amount of nodes, activation
model.add(Dense(256,activation='relu'))

# place a dropout layer
# 0.5 drop out rate is recommended, half input nodes will be dropped at each update
model.add(Dropout(0.5))

# defining the ouput layer of our network
model.add(Dense(43,activation='softmax'))

model.summary()

model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train_processed,
                    y_train,
                    batch_size=500,
                    epochs=50,
                    verbose=1,
                    validation_data=(X_validation_processed,y_validation))

score = model.evaluate(X_test_processed,y_test)
print('Test accuracy',score[1])

history.history.keys()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Training and validation losses')
plt.xlabel('epochs')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epochs')

prediction = model.predict_classes(X_test_processed)
y_true_label= y_test

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_true_label,prediction)
plt.figure(figsize=(20,20))
sns.heatmap(matrix,annot=True)

L=6
W=6
fig , axes = plt.subplots(L,W,figsize=(12,12))
axes = axes.ravel()
for i in range(0,L*W):
  axes[i].imshow(X_test[i])
  axes[i].set_title('Prediction ={}\n True={}'.format(prediction[i],y_true_label[i]))
  axes[i].axis('off')
plt.subplots_adjust(wspace=1)

model.save('my_model.h5')
