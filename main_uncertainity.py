import pickle
import keras
import numpy as np
from keras.applications.xception import Xception
from keras.layers import Input, Dense, Flatten, Model
from functions import DataAugment
from keras import backend as K

# uncertainity loss function


def uncertainty_loss(variance):
    def mae_loss(y_true, y_pred):
        temp = K.abs(y_true-y_pred)*K.exp(-variance)
        loss = K.mean(temp)
        loss = loss + 5.0*K.mean(K.abs(variance))
        return loss

    return mae_loss

# Generator function defined for data augmentation


def Generator(x_train, gender_train, y_train, batch_size):
    loopcount = len(y_train)//batch_size
    i = 0
    while (True):
        if i > loopcount:
            i = 0
        # i=np.random.randint(0,loopcount)
        x_train_batch = x_train[i*batch_size:(i+1)*batch_size, :, :, :]
        x_train_batch = DataAugment(x_train_batch)
        gender_train_batch = gender_train[i*batch_size:(i+1)*batch_size]
        y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
        inputs = [x_train_batch, gender_train_batch]
        target = y_train_batch
        yield (inputs, target)
        i = i+1


batch_size = 16
epochs = 30

# loading the data
with open('dataR2.pkl', 'rb') as f:
    dataR2 = pickle.load(f)

with open('dataR1.pkl', 'rb') as f:
    dataR1 = pickle.load(f)

with open('dataHand.pkl', 'rb') as f:
    dataHand = pickle.load(f)

with open('data_age.pkl', 'rb') as f:
    age = pickle.load(f)

with open('data_gender.pkl', 'rb') as f:
    gender = pickle.load(f)

# converting the data into arrays

data = np.asarray(dataHand, dtype=np.float32)
dataR1 = np.asarray(dataR1, dtype=np.float32)
dataR2 = np.asarray(dataR2, dtype=np.float32)
data[:, :, :, 1] = dataR1[:, :, :, 1]
data[:, :, :, 2] = dataR2[:, :, :, 2]
print(data.shape)

age = np.asarray(age)
gender = 2*(np.asarray(gender) - 0.5)
data /= 255

# shuffling images

x_final = []
y_final = []
gender_final = []

random_no = np.random.choice(data.shape[0], size=data.shape[0], replace=False)
for i in random_no:
    x_final.append(data[i, :, :, :])
    y_final.append(age[i])
    gender_final.append(gender[i])

# converting the lists to numpy arrays

x_final = np.asarray(x_final)
y_final = np.asarray(y_final)
gender_final = np.asarray(gender_final)

# print(y_final[:50])
# print(gender_final[:50])

# splitting the data into test, train and validation
k = 500  # Decides split count
x_test = x_final[:k, :, :, :]
y_test = y_final[:k]
gender_test = gender_final[:k]

x_valid = x_final[k:2*k, :, :, :]
y_valid = y_final[k:2*k]
gender_valid = gender_final[k:2*k]

x_train = x_final[2*k:, :, :, :]
y_train = y_final[2*k:]
gender_train = gender_final[2*k:]

# print ('x_train shape:'+ str(x_train.shape))
# print ('y_train shape:'+ str(y_train.shape))
# print ('gender_train shape:'+ str(gender_train.shape))
# print ('x_valid shape:'+ str(x_valid.shape))
# print ('y_valid shape:'+ str(y_valid.shape))
# print ('gender_valid shape:' + str(gender_valid.shape))
# print ('x_test shape:'+ str(x_test.shape))
# print ('y_test shape:'+ str(y_test.shape))

# defining the basemodel
base_model = Xception(weights='imagenet', include_top=False)
# used in the reserach paper
# base_model = Xception(weights='imagenet', include_top=False)
input_x = Input(shape=(560, 560, 3), name='input1')
input_gender = Input(shape=(1,), dtype='float32', name='input2')

# stacking the model layers
layer1 = base_model(input_x)
gender_embedding = Dense(32)(input_gender)
layer2 = keras.layers.Conv2D(256, kernel_size=(3, 3))(layer1)
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

Embedding = keras.layers.Conv2D(256, kernel_size=(2, 2), strides=1)(layer1)
Embedding = keras.layers.AveragePooling2D(pool_size=(9, 9))(Embedding)
Embedding = Flatten()(Embedding)
print(K.int_shape(Embedding))
variance = Dense(1)(Embedding)

model = Model(inputs=[input, input_gender], outputs=predictions)

# printing the layers of the model
# for i,layer in enumerate(model.layers):
#     print (i,layer.name)

# training for 60 epochs  at 0.0003 lr
Adam = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss=uncertainty_loss(variance), metrics=['MAE'])

checkpoint =keras.callbacks.ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',save_weights_only=True,period=30)
history=model.fit([x_train,gender_train],y_train,batch_size=batch_size,epochs=60,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

##Visulization
weights=model.layers[-1].get_weights()[0]
print (weights.shape)

# training for 30 epochs at lr=0.0001
Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss=uncertainty_loss(variance), metrics=['MAE'])
history = model.fit([x_train, gender_train], y_train, batch_size=batch_size, epochs=30,
                    verbose=1, validation_data=([x_valid, gender_valid], y_valid), callbacks=[checkpoint])
score = model.evaluate([x_test, gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

# training for 30 epochs at lr=0.0001
Adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss=uncertainty_loss(variance), metrics=['MAE'])
history = model.fit([x_train, gender_train], y_train, batch_size=batch_size, epochs=20,
                    verbose=1, validation_data=([x_valid, gender_valid], y_valid), callbacks=[checkpoint])
score = model.evaluate([x_test, gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

# saving model weights
model.save_weights("model.h5")
with open('history.pkl', 'wb') as f:
	pickle.dump(history.history, f)
