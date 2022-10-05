import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import Input, Dense, Flatten, Model
from keras import backend as K
import pickle
import numpy as np


batch_size = 32
epochs = 30

# Loading the data
# Hand data
with open(".data/dataHand.pkl", "rb") as f:
    x = pickle.load(f)

# Age data
with open(".data/data_age.pkl", "rb") as f:
    y = pickle.load(f)

# Gender data
with open(".data/data_gender.pkl", "rb") as f:
    gender = pickle.load(f)

# converting data to arraydata
x = (np.asarray(x, dtype=np.float32))/255
y = np.asarray(y)
gender = 2*(np.asarray(gender) - 0.5)

x_final = []
y_final = []
gender_final = []

# shuffling the data 
random_no = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(random_no)

for i in random_no:
    x_final.append(x[i, :, :, :])
    y_final.append(y[i])
    gender_final.append(gender[i])

# converting the python list to numpy array
x_final = np.asarray(x_final)
y_final = np.asarray(y_final)
gender_final = np.asarray(gender_final)

# splitting the data into train, val 
k = 500 # splitcount

# train split
x_train = x_final[2*k:, :, :, :]
y_train = y_final[2*k:]
gender_train = gender_final[2*k:]

# test split
x_test = x_final[:k, :, :, :]
y_test = y_final[:k]
gender_test = gender_final[:k]

# validation split
x_valid = x_final[k:2*k, :, :, :]
y_valid = y_final[k:2*k]
gender_valid = gender_final[k:2*k]

# defining the base model using pretrained weights from imagenet
base_model = InceptionV3(weights='imagenet', include_top=False)  
# used in the reserach paper
# base_model = Xception(weights='imagenet', include_top=False) 
input_x = Input(shape=(480, 480, 3), name='input1')
input_gender = Input(shape=(1,), dtype='float32', name='input2')

# stacking the model layers
layer1 = base_model(input_x)
gender_embedding = Dense(16)(input_gender)
layer2 = keras.layers.Conv2D(256, kernel_size=(1, 1))(layer1)
print(K.int_shape(layer1))

layer3 = keras.layers.MaxPooling2D(pool_size=(3, 3))(layer2)
print(K.int_shape(layer2))

# printing the layers of the base model 
# for i,layer in enumerate(base_model.layers):
#     print (i,layer.name)

layer3 = Flatten()(layer3)
x = keras.layers.Concatenate(axis=1)([layer3, gender_embedding])
print(K.int_shape(x))

predictions = Dense(1)(x)

model = Model(inputs=[input_x,input_gender], outputs=predictions)

# printing the layers of the model
# for i,layer in enumerate(model.layers):
#     print (i,layer.name)

# training for 60 epochs at lr = 0.0003
Adam = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

# Save weights after every epoch
checkpoint = keras.callbacks.ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True, period=30)
history = model.fit([x_train, gender_train], y_train, batch_size=batch_size, epochs=50,
                    verbose=1, validation_data=([x_valid, gender_valid], y_valid), callbacks=[checkpoint])
score = model.evaluate([x_test, gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

##Visulization
weights = model.layers[-1].get_weights()[0]
print(weights.shape)

# training for 30 epochs at lr=0.0001
Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train, gender_train], y_train, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=([x_valid, gender_valid], y_valid), callbacks=[checkpoint])
score = model.evaluate([x_test, gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

# ṭraining for 30 epochs at lr= 0.00001
Adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train, gender_train], y_train, batch_size=batch_size, epochs=20,
                    verbose=1, validation_data=([x_valid, gender_valid], y_valid), callbacks=[checkpoint])
score = model.evaluate([x_test, gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

model.save_weights("model.h5")
with open('history.pkl', 'wb') as f:
	pickle.dump(history.history, f)

