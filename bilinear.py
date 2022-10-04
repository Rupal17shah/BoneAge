import keras
from keras.applications.inception_v3 import InceptionV3
from keras.model import Model
from keras.layers import Flatten, Dense, Reshape, Input, Conv2D
import numpy as np
import pickle

batch_size = 32
epochs = 30
split_size = 500

def matmul(x):
    return x[0]*x[1]*x[2]

with open('./data/data.pkl', 'rb') as f1, open('./data/data_age.pkl', 'rb') as f2, open('./data/data_gender.pkl', 'rb') as f3:
    x = pickle.load(f1)
    y = pickle.load(f2)
    gender = pickle.load(f3)

x = (np.asarray(x, dtype=np.float16))/255
y = np.asarray(y)
gender = 2*(np.asarray(gender)-0.5)

finalX = []
finalY = []
finalGender = []

rand_num = np.random.choice(x.shape[0], size=x.shape[0], replace=False)

for i in rand_num:
    finalX.append(x[i,:,:,:])
    finalY.append(y[i])
    finalGender.append(gender[i])

x_final = np.array(finalX)
y_final = np.array(finalY)
gender_final = np.array(finalGender)

x_train = x_final[2*split_size:,:,:,:]
y_train = y_final[2*split_size:]
gender_train = gender_final[2*split_size:]
x_valid = x_final[split_size:2*split_size,:,:,:]
y_valid = y_final[split_size:2*split_size]
gender_valid = gender_final[split_size:2*split_size]
x_test = x_final[:split_size,:,:,:]
y_test = y_final[:split_size]
gender_test = gender_final[:split_size]


base = InceptionV3(weights='imagenet', include_top=False)
input = Input(shape(800,600,3), name='in1')
input_gender = Input(shape(1,), name='in2')
gender_emb = Dense(16)(input_gender)


x = base(input)
x = Conv2D(512, kernal_size(1,1))(x)
x = Conv2D(128, kernal_size(1,1))(x)
x = Reshape((391,128))(x)
x = keras.layers.dot(inputs=[x,x],axes=1)
x = Flatten()(x)
final = keras.layers.Concatenate(axis=1)([x,gender_emb])
predictions = Dense(1)(final)

model = Model(inputs=[input,input_gender], outputs=predictions)

Adam = keras.optimizers.Adam(lr=0.0003,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

checkpoint = keras.callbacks.ModelCheckpoint(filepath='./weights/weights-bilinear.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True, period=30)

history = model.fit([x_train,gender_train],y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)

for layer in base.layers[:250]:
    layer.trainable=False

Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

history = model.fit([x_train,gender_train], y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)

for layer in base.layers[:310]:
    layer.trainable=False

Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

history = model.fit([x_train,gender_train], y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)

model.save_weights("model-bilinear.h5")
with open('./data/history-bilinear.pkl', 'wb') as f:
	pickle.dump(history.history, f)