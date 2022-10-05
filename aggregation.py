from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import pickle
import numpy as np
import keras
from functions import *
import os

batch_size = 16
#Load data
f = open('dataR2.pkl', 'rb')
dataR2 = pickle.load(f)
f.close()

f = open('dataR1.pkl', 'rb')
dataR1 = pickle.load(f)
f.close()

f = open('dataHand.pkl', 'rb')
dataHand = pickle.load(f)
f.close()

f = open('data_age.pkl', 'rb')
age = pickle.load(f)
f.close()

f = open('data_gender.pkl','rb')
gender = pickle.load(f)
f.close()

data = np.asarray(dataHand, dtype=np.float32)/255.
dataR1 = np.asarray(dataR1, dtype=np.float32)
dataR2 = np.asarray(dataR2, dtype=np.float32)
data[:,:,:,1] = dataR1[:,:,:,1]
data[:,:,:,2] = dataR2[:,:,:,2]
print (data.shape)
age = np.asarray(age)
gender = np.asarray(gender)

gender =2*( gender-0.5) #setting -1 and 1 instead of 0 and 1
x_final = []
y_final = []
gender_final = []

random_no = np.arange(age.shape[0])
np.random.seed(0) #random number initialsied from 0
np.random.shuffle(random_no) #shuffles the array of random_no
for i in random_no:
    x_final.append(data[i,:,:,:])
    y_final.append(age[i])
    gender_final.append(gender[i])


x_final = np.asarray(x_final)
y_final = np.asarray(y_final)
gender_final = np.asarray(gender_final)
split_count= 500
x_train = x_final[2*split_count:,:,:,:]
y_train = y_final[2*split_count:]
gender_train = gender_final[2*split_count:]
x_valid = x_final[split_count:2*split_count,:,:,:]
y_valid = y_final[split_count:2*split_count]
gender_valid = gender_final[split_count:2*split_count]
x_test = x_final[:split_count,:,:,:]
y_test = y_final[:split_count]
gender_test = gender_final[:split_count]
#helps us get training,vaidation and test dataset

del data
del dataR1
del dataR2
del x_final
#No longer required
#y_test = keras.utils.to_categorical(y_test,240)
#y_train = keras.utils.to_categorical(y_train,240)
#y_valid = keras.utils.to_categorical(y_valid,240)
#y_train = softlabel(y_train,240)
#y_valid = softlabel(y_valid,240)
#y_test = softlabel(y_test,240)

base_model = Xception(weights='imagenet', include_top=False) #input size is different
for i,layer in enumerate(base_model.layers):
    print (i,layer.name) #returns the architechture of Xception
input = Input(shape=(560,560,3),name='input1')
input_gender = Input(shape=(1,),dtype='float32',name='input2')
output = base_model(input)
gender_embedding=Dense(32)(input_gender)
x = keras.layers.Conv2D(256,kernel_size=(3,3))(output)
#print (K.int_shape(output))
x = keras.layers.MaxPooling2D(pool_size=(3,3))(x)
#print (K.int_shape(x))
x=Flatten()(x)
f = keras.layers.Concatenate(axis=1)([x,gender_embedding])
#print (K.int_shape(f)) 
predictions = Dense(1)(f)
model = Model(inputs=[input,input_gender], outputs=predictions)
for i,layer in enumerate(model.layers):
    print (i,layer.name) #returns the final architechture

Adam=keras.optimizers.Adam(lr=0.0003,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

DataGen = ImageDataGenerator(rotation_range=20,width_shift_range=0.15,height_shift_range=0.15,zoom_range=0.2,horizontal_flip=True)

checkpoint =keras.callbacks.ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',save_weights_only=True,period=30)
history = model.fit_generator(DataGen.flow([x_train,gender_train],y_train,batch_size=batch_size),steps_per_epoch=np.ceil(len(y_train)/batch_size),epochs=350,verbose=1,validation_data=([x_valid,gender_valid],y_valid))
history=model.fit([x_train,gender_train],y_train,batch_size=batch_size,epochs=80,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

# weights=model.layers[-1].get_weights()[0]
# print (weights.shape)

Adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train,gender_train], y_train,batch_size=batch_size,epochs=30,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train,gender_train],y_train,batch_size=batch_size,epochs=20,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

#as we train with 3 different learning rates.

#saving the model
model.save_weights("model.h5")
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
f.close()