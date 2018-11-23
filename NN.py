import numpy as np
import tensorflow as tf
import pandas as pd
import h5py
import random
import matplotlib.pyplot as plt
import cv2
import time 
import keras.backend as K
import keras 
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras.losses import mean_absolute_error, categorical_crossentropy,mean_absolute_error
from keras.layers import Flatten, Dropout, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, concatenate
from keras.models import Input,Model
from keras.callbacks import TensorBoard,ModelCheckpoint

TRAINING_SIZE = 100000
TESTING_SIZE = 2000
VALIDATION_SIZE = 2000

TOTAL_IMAGES = 180000
IMAGE_SIZE = 224

BATCH_SIZE = 8
TRIPLET_INDEX = 0

ENCODINGS_DIM = 1000


arod = h5py.File('./AROD_HDF/AROD.hdf','r')
triplets = pd.read_csv('./triplets.csv').get_values()[0:TRAINING_SIZE]
training_set = triplets[:,1:4]


def get_triplet():
    global TRIPLET_INDEX
    triplet = training_set[TRIPLET_INDEX]
    
    a = arod['IMAGES'][triplet[0]]
    p = arod['IMAGES'][triplet[1]]
    n = arod['IMAGES'][triplet[2]]
    
    sa = arod['SCORES'][triplet[0]][0]        
    sp = arod['SCORES'][triplet[1]][0]        
    sn = arod['SCORES'][triplet[2]][0]        
    TRIPLET_INDEX = TRIPLET_INDEX + 1
    if TRIPLET_INDEX > 80000:
        TRIPLET_INDEX = 0 
    return a, p, n, sa, sp, sn 



def Generate():
    while True:
        list_a = []
        list_p = []
        list_n = []
        label = []

        for i in range(BATCH_SIZE):
            a, p, n, sa, sp, sn = get_triplet()
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)
            label.append([sa,sn])
            
        A = preprocess_input(np.array(list_a, dtype = 'float32'))
        B = preprocess_input(np.array(list_p, dtype = 'float32'))
        C = preprocess_input(np.array(list_n, dtype = 'float32'))
        label = np.array(label,dtype = 'float32')
        yield [A, B, C], label



train_generator = Generate()
batch = next(train_generator)
############## LOSS Function ########################### 
def identity_loss(y_true, y_pred):
    r = y_true[0] - y_pred[0]
    #return K.mean(y_pred - 0 * y_true)
    return K.sum(y_pred - 0 * y_true,axis=-1)

def Le(X):
    a, p, n = X
    m = 0.2 
    loss = K.relu(m + K.sum(K.square(a-p),axis=-1,keepdims=True) - K.sum(K.square(a-n),axis=-1,keepdims=True))
    return loss

def Ld_1(X):
    a, p, n = X
    m = 0.3
    loss = K.relu(m+ K.sqrt(K.sum(K.square(a),axis=-1,keepdims=True)) - K.sqrt(K.sum(K.square(n),axis=-1,keepdims=True)))
    return loss

def triplet_loss(y_true,y_pred):
    sa = y_true[0]
    sp = y_true[1]
    sn = y_true[2]
    
    ld = y_pred[0]
    le = y_pred[1]
    
    return (sn - sa)*ld + le



def GetBaseModel():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    dense_1 = Dense(ENCODINGS_DIM)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model

def GetModel(base_model):
    input_1 = Input((IMAGE_SIZE,IMAGE_SIZE,3))
    input_2 = Input((IMAGE_SIZE,IMAGE_SIZE,3))
    input_3 = Input((IMAGE_SIZE,IMAGE_SIZE,3))

    r1 = base_model(input_1)
    r2 = base_model(input_2)
    r3= base_model(input_3)

    loss_le = Lambda(Le)([r1,r2,r3])
    loss_ld1 = Lambda(Ld_1)([r1,r2,r3])
    loss = concatenate([loss_le,loss_ld1],axis=-1)
    
    
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)    
    model.compile(loss=identity_loss, optimizer=Adam())#0.000003
    return model
model = GetModel(GetBaseModel())
model.summary()


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')  #   
tb = TensorBoard(log_dir='./logs/', write_graph=True)
#tb = TensorBoard(log_dir='./log/', histogram_freq=5, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True) 
callbacks_list = [checkpoint, tb]
history = model.fit_generator(train_generator,
                    epochs=100, 
                    verbose=1, 
                    workers=20,
                    steps_per_epoch=5000,
                    validation_steps=100,
                    callbacks=callbacks_list)

model.save_weights(filepath='last.h5py')




