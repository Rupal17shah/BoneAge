import numpy as np
import cv2
import os
import pandas as pd
from six.moves import cPickle

train_dir = '../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset'

X_train = []
y_age = []
y_gender = []

df = pd.read_csv('../input/rsna-bone-age/boneage-training-dataset.csv')
print(df.head(10))

#print(df.dtypes)


print ('Loading data set...')

for i in os.listdir('../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset'):
    #here i will be the name of the images in the folder
    #i[:-4] helps us get rid of .png by using reverse slicing
    #We append to the empty list the ages of the photos whose name is present in the id
    y_age.append(df.boneage[df.id == int(i[:-4])].tolist()[0])
    #Here we convert false true to 0 and 1 and append the same to a list
    a = df.male[df.id == int(i[:-4])].tolist()[0]
    if a:
        y_gender.append(1)
    else:
        y_gender.append(0)
    img = cv2.imread('../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'+i)
#     print (img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(300,300))
    x = np.asarray(img, dtype=np.uint8)
    X_train.append(x)
    #a list of images as numpy arrays.
print ('100% completed loading data')
#Now we have all the 3 attributes i.e image, gender and boneage with correspondence maintained.
#i.e. first image in X, first gender and first boneage match
#Save data
train_pkl = open('data.pkl','wb')
cPickle.dump(X_train, train_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_pkl.close()

train_age_pkl = open('data_age.pkl','wb')
cPickle.dump(y_age, train_age_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_age_pkl.close()

train_gender_pkl = open('data_gender.pkl','wb')
cPickle.dump(y_gender, train_gender_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_gender_pkl.close()