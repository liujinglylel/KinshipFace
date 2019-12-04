'''
Author: Jing Liu
Date: 2019/12/3
'''

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import os, glob
from imageio import imread
from scipy.spatial import distance
from random import choice, sample
import sys
from sklearn.model_selection import train_test_split

# preprocess train data
filepath = sys.argv[1]
os.chdir(filepath + "train//")
all_ppl = [x.split("\\")[0] + '/' + x.split("\\")[1] for x in glob.glob("*/*")]

match = []
train_df = pd.read_csv(filepath + "train_relationships.csv")
train_np = train_df.to_numpy()
for i in range(train_np.shape[0]):
	if train_np[i][0] in all_ppl and train_np[i][1] in all_ppl:
		match.append((train_np[i][0], train_np[i][1]))

X_related = []
for item in match:
	for it in range(10):
		f1 = glob.glob(item[0])[0]
		f2 = glob.glob(item[1])[0]
		if f1 is not None and f2 is not None:
			if len(os.listdir(f1)) > 0 and len(os.listdir(f2)) > 0:
				file1 = random.choice(os.listdir(f1))
				file2 = random.choice(os.listdir(f2))
				X_related.append((f1+'/'+file1,f2+'/'+file2))
X_unrelated = []
count = 1
while count <= len(X_related):
	p = random.choices(all_ppl, k=2)
	p1 = p[0]
	p2 = p[1]
	if p1 != p2 and (p1,p2) not in match and (p2,p1) not in match:
		f1 = glob.glob(p1)[0]
		f2 = glob.glob(p2)[0]
		if f1 is not None and f2 is not None:
			if len(os.listdir(f1)) > 0 and len(os.listdir(f2)) > 0:
				file1 = random.choice(os.listdir(f1))
				file2 = random.choice(os.listdir(f2))
				X_unrelated.append((f1+'/'+file1,f2+'/'+file2))
				count += 1

all_images = list(set([x[0] for x in (X_related + X_unrelated)] + [x[1] for x in (X_related + X_unrelated)]))

'''
Image read function for this pretained VGG Model resnet50
'''
def read_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


'''
To run VGG model, it will be more ideal to fit in batch mode
The following function serves as a generator for train images
'''
def gen(related, unrelated, batch_size=16):
    while True:
        batch_tuples_related = sample(related, batch_size // 2)
        batch_tuples_unrelated = sample(unrelated, batch_size // 2)
        batch_tuples = batch_tuples_related + batch_tuples_unrelated
        labels = [1] * len(batch_tuples_related) + [0] * len(batch_tuples_unrelated)
        X1 = np.array([read_img(x[0]) for x in batch_tuples])
        X2 = np.array([read_img(x[1]) for x in batch_tuples])
        yield [X1, X2], labels


'''
This function will generate a CNN model which is based on VGG resNet50 model
We will apply some additional layers after we get the initial representation from VGG
'''
def create_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))
    vgg_model = VGGFace(model='resnet50', include_top=False)
    for x in vgg_model.layers[:-3]:
        x.trainable = True
    x1 = vgg_model(input_1)
    x2 = vgg_model(input_2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    x3 = Subtract()([x1, x2])
    x = Multiply()([x3, x3])
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([input_1, input_2], out)   
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    return model


model_path = "../kinship_vgg_face.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)
callbacks_list = [checkpoint, reduce_on_plateau]
model = create_model()
train_related, val_related = train_test_split(X_related, test_size=0.3)
train_unrelated, val_unrelated = train_test_split(X_unrelated, test_size=0.3)

model.fit_generator(gen(train_related, train_unrelated, batch_size=16),
                    validation_data=gen(val_related, val_unrelated, batch_size=16), epochs=200, verbose=1,
                    workers = 8, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

os.chdir(filepath)
submission = pd.read_csv('sample_submission.csv')
test_np = submission.to_numpy()
test_pair = [(x[0].split("-")[0], x[0].split("-")[1]) for x in test_np]
test_images = list(set([x[0] for x in test_pair] + [x[1] for x in test_pair]))

os.chdir(filepath + "test//")
prediction = []
for item in test_pair:
    X1 = np.array([read_img(item[0])])
    X2 = np.array([read_img(item[1])])
    pred = model.predict([X1, X2])
    prediction.append(pred[0][0])

os.chdir(filepath)
sub_df = pd.read_csv('sample_submission.csv')
sub_df.is_related = prediction
sub_df.to_csv("submission_vggmodel.csv", index=False)
