from zipfile import ZipFile
file_name="C:\\Users\\intel\\OneDrive\\Desktop\\covid\\CovidDataset.zip"
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('done')

from keras.layers import *
from keras.models import *
import keras as tf

# trainig model
model=Sequential() #create a blank model
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Dropout(0.25)) #reduces the boverfiting

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu')) #hidden layer
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid')) #output layer

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#moduling train images
from tensorflow.keras.preprocessing import image
train_datagen=image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=image.ImageDataGenerator(rescale=1./255)

# reshaping test and validation images
train_generator=train_datagen.flow_from_directory('/content/CovidDataset/Train',target_size=(224,224),batch_size=32,class_mode='binary')
validation_generator=test_datagen.flow_from_directory('/content/CovidDataset/Val',target_size=(224,224),batch_size=32,class_mode='binary')

#train the model
history=model.fit(train_generator,steps_per_epoch=7,epochs=20,validation_data=validation_generator,validation_steps=1)

from keras.preprocessing import image
import numpy as np
img=image.load_img('/content/CovidDataset/Val/Covid/4-x-day1.jpg',target_size=(224,224))
img=image.img_to_array(img)
img=img/255
img=np.expand_dims(img,axis=0)
pred=model.predict(img)


if pred[0][0]>0.5:
  print("covid negative")
else:
  print("covid positive")