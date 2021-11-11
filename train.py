# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:35:38 2021

@author: s1253
"""

import numpy as np
import sklearn.preprocessing as preprocessing

from sklearn.model_selection import train_test_split
from util import load_dataframe
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, BatchNormalization,\
   MaxPooling2D, AveragePooling2D
from tensorflow.keras import Model

gen_folders = {    
    "gen01_red-blue" : 151,
    "gen01_red-green" : 151,
    "gen01_yellow" : 151,
    "gen02_crystal" : 251,
    "gen02_gold" : 251,
    "gen02_silver" : 251,
    "gen03_emerald" : 386,
    "gen03_firered-leafgreen" : 151,
    "gen03_ruby-sapphire" : 386,
    "gen04_diamond-pearl" : 493,
    "gen04_heartgold-soulsilver" : 386,
    "gen04_platinum" : 386,
    "gen05_black-white" : 649
}

#load data and split
df = load_dataframe(gen_folders)

df_train, df_test = train_test_split(df,test_size=0.3, random_state = 111)

def get_xy(df):
    rows = df.values.shape[0]
    arr_x = np.zeros((rows,64,64,3))
    arr_y = df["type1"].values.reshape(-1,1)
    
    for i in range(0,rows):
        arr_x[i] = df["sprite"].iloc[i]
    
    y_domain = np.array(range(1,19)).reshape(-1,1)
    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(y_domain)
    arr_y = oh_encoder.transform(arr_y).toarray()
    
    return arr_x, arr_y

x_train, y_train = get_xy(df_train)
x_test, y_test = get_xy(df_test)



''' training '''
input_layer = Input((64,64,3))

x =Conv2D(filters=32, kernel_size=7, strides=1, padding="same", activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x =Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation='relu')(x)
x =Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x =Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')(x)
x =Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x =Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')(x)
x =Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')(x)
x = AveragePooling2D((8, 8), strides=(8, 8))(x)

x = Flatten()(x)

x = Dense(64, activation="relu")(x)

x =  Dense(18)(x)

out_put = Activation('softmax')(x)

model = Model(input_layer, out_put)

model.summary()


filepath = "model/after_data_augumentation.h5"
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint

# save model
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')

#History = model.fit(x_train, y_train, batch_size=32, validation_data = (x_test,y_test), epochs = 30, callbacks=[checkpoint])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data generate

img_gen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )



History = model.fit_generator(img_gen.flow(x_train, y_train, batch_size = 32),
                                      steps_per_epoch = len(x_train)/32, validation_data = (x_test,y_test), epochs = 30 , callbacks=[checkpoint])
                                      
#%%


import matplotlib.pyplot as plt

# visualize loss and accuracy

arr = History.history['accuracy'][-1]
val = History.history['val_accuracy'][-1]


plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig(r"accracy_loss/data_aug.png")

plt.show()
#%%

model.load_weights("model/original.h5")


# visualize the results

CLASSES = np.array(['normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost'\
                    , 'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark', 'fairy'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]


fig = plt.figure(figsize=(12, 9))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


for i in range(10):
    indices = np.random.choice(range(len(x_test)), 18)
    for j, idx in enumerate(indices):
        img = x_test[idx]
        ax = plt.subplot(3, 6, j+1)
        ax.axis('off')
        ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=15, ha='center', transform=ax.transAxes) 
        ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=15, ha='center', transform=ax.transAxes)
        ax.imshow(img)
        
    #plt.savefig("result/pictur_{}".format(i+1))
    plt.clf()

















