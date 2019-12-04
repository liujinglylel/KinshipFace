'''
Author: Jing Liu
Date: 2019/12/3
'''

import sys
from sklearn.model_selection import train_test_split
import os, glob
import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from skimage import color
from imageio import imread
import csv
from keras.layers import ZeroPadding2D, Convolution2D, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from skimage.transform import resize
import cv2
from keras.models import load_model
import lightgbm as lgb

filepath = sys.argv[1]

# Preprocess training data
os.chdir(filepath + "train//")
all_ppl = [x.split("\\")[0] + '/' + x.split("\\")[1] for x in glob.glob("*/*")]

match = []
train_df = pd.read_csv(filepath + "train_relationships.csv")
train_np = train_df.to_numpy()
for i in range(train_np.shape[0]):
	if train_np[i][0] in all_ppl and train_np[i][1] in all_ppl:
		match.append((train_np[i][0], train_np[i][1]))

# Generate train image pairs which are related
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

# Generate train image pairs which are not related -- same length of related pairs
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

######################
# Autodecoder Method #
######################
'''
create an autoencoder model to encode images
'''
def create_autoencoder():
    input_img = Input(shape=(224, 224, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder


os.chdir(filepath + "train//")
autoencoder = create_autoencoder()
new_faces = []
for item in all_images:
    new_faces.append(color.rgb2gray(imread(item)).reshape(224,224,1))
np_new_faces = np.asarray(new_faces)
X_train, X_test = train_test_split(np_new_faces, test_size=0.3)
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))

# Save autoencoder distance to csv files
auto_store = {}
decoded_images = autoencoder.predict(np_new_faces)
for i in range(len(all_images)):
    auto_store[all_images[i]] = decoded_images[i]
auto_related = []
for pair in X_related:
    pre_1 = auto_store[pair[0]].flatten()
    pre_2 = auto_store[pair[1]].flatten()
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    auto_related.append((eu_dist, co_dist))

auto_unrelated = []
for pair in X_unrelated:
    pre_1 = auto_store[pair[0]].flatten()
    pre_2 = auto_store[pair[1]].flatten()
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    auto_unrelated.append((eu_dist, co_dist))

os.chdir(filepath)
with open("auto_dist_related.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(auto_related)

with open("auto_dist_unrelated.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(auto_unrelated)

# run autoencoder on test images and save results
os.chdir(filepath + "test//")
test_df = pd.read_csv(filepath + "sample_submission.csv")
test_np = test_df.to_numpy()
test_pair = [(x[0].split("-")[0], x[0].split("-")[1]) for x in test_np]
test_images = list(set([x[0] for x in test_pair] + [x[1] for x in test_pair]))
test_faces = []
for item in test_images:
    test_faces.append(color.rgb2gray(imread(item)).reshape(224,224,1))
np_test_faces = np.array(test_faces)

test_decoded_imgs = autoencoder.predict(np_test_faces)
test_store = {}
for i in range(len(test_images)):
    test_store[test_images[i]] = test_decoded_imgs[i]
auto_test = []
for pair in test_pair:
    pre_1 = test_store[pair[0]].flatten()
    pre_2 = test_store[pair[1]].flatten()
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    auto_test.append((eu_dist, co_dist))

os.chdir(filepath)
with open("auto_dist_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(auto_test)


##################
# VGGFace Method #
##################
'''
create a VGGModel
'''
def create_vgg_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights('vgg_face_weights.h5')
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_descriptor


vgg_face_descriptor = create_vgg_model()
'''
read image for VGG Model
'''
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Save VGG distance to csv files
os.chdir(filepath + "train//")
vgg_store = {}
for i in range(len(all_images)):
	item = all_images[i]
	img_pre = vgg_face_descriptor.predict(preprocess_image(item))[0,:]
	vgg_store[item] = img_pre

vgg_dist_related = []
for item in X_related:
	pre_1 = vgg_store[item[0]]
	pre_2 = vgg_store[item[1]]
	eu_dist = distance.euclidean(pre_1, pre_2)
	co_dist = distance.cosine(pre_1, pre_2)
	vgg_dist_related.append((eu_dist, co_dist))

vgg_dist_unrelated = []
for item in X_unrelated:
	pre_1 = vgg_store[item[0]]
	pre_2 = vgg_store[item[1]]
	eu_dist = distance.euclidean(pre_1, pre_2)
	co_dist = distance.cosine(pre_1, pre_2)
	vgg_dist_unrelated.append((eu_dist, co_dist))

os.chdir(filepath)
with open("vgg_dist_related.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(vgg_dist_related)

with open("vgg_dist_unrelated.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(vgg_dist_unrelated)

# run VGG model on test images and save results
os.chdir(filepath + "test//")
vgg_test = {}
for i in range(len(test_images)):
    item = test_images[i]
    img_pre = vgg_face_descriptor.predict(preprocess_image(item))[0,:]
    vgg_test[item] = img_pre
vgg_test_dist = []
for pair in test_pair:
    pre_1 = vgg_test[pair[0]].flatten()
    pre_2 = vgg_test[pair[1]].flatten()
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    vgg_test_dist.append((eu_dist, co_dist))

os.chdir(filepath)
with open("vgg_dist_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(vgg_test_dist)


##################
# Facenet Method #
##################
'''
This pretained facenet model requires input size as 160x160, this function is to align images to the required size
'''
def load_and_align_images(filepaths, margin, image_size):
    aligned_imgs = []
    count = 1
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned_imgs.append(aligned)
        count += 1
    return np.array(aligned_imgs)


# Save facenet distances to csv
os.chdir(filepath + "train//")
aligned_images = load_and_align_images(all_images, 0, 160)
os.chdir(filepath)
model = load_model('facenet_keras.h5')
facenet_presentation = model.predict(aligned_images)
facenet_store = {}
for i in range(len(all_images)):
    facenet_store[all_images[i]] = facenet_presentation[i]

facenet_dist_related = []
for pair in X_related:
    pre_1 = facenet_store[pair[0]]
    pre_2 = facenet_store[pair[1]]
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    facenet_dist_related.append((eu_dist, co_dist))
facenet_dist_unrelated = []
for pair in X_unrelated:
    pre_1 = facenet_store[pair[0]]
    pre_2 = facenet_store[pair[1]]
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    facenet_dist_unrelated.append((eu_dist, co_dist))

with open("facenet_dist_related.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(facenet_dist_related)
with open("facenet_dist_unrelated.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(facenet_dist_unrelated)

# Run Facenet model on test images and save results
os.chdir(filepath + "test//")
aligned_test_images = load_and_align_images(test_images, 0, 160)
facenet_test_presentation = model.predict(aligned_test_images)
facenet_test_store = {}
for i in range(len(test_images)):
    facenet_test_store[test_images[i]] = facenet_test_presentation[i]
facenet_test_dist = []
for pair in test_pair:
    pre_1 = facenet_test_store[pair[0]]
    pre_2 = facenet_test_store[pair[1]]
    eu_dist = distance.euclidean(pre_1, pre_2)
    co_dist = distance.cosine(pre_1, pre_2)
    facenet_test_dist.append((eu_dist, co_dist))

os.chdir(filepath)
with open("facenet_dist_test.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerows(facenet_test_dist)


############
# Lightgbm #
############
'''
Now we have six types of distances from three different models.
The next step is to fit a decision tree based on these distances.
The algorithm I choose here is Lightgbm from Microsoft
'''
related_df_1 = pd.read_csv('auto_dist_related_v2.csv', names=["auto_euclidean", "auto_cosine"])
related_df_2 = pd.read_csv('facenet_dist_related.csv', names=["facenet_euclidean", "facenet_cosine"])
related_df_3 = pd.read_csv('vgg_dist_related.csv', names=["vgg_euclidean", "vgg_cosine"])
related_df = pd.concat([related_df_1, related_df_2, related_df_3], axis=1, sort=False)
related_label = [1] * related_df.shape[0]
related_df['label'] = related_label

unrelated_df_1 = pd.read_csv('auto_dist_unrelated_v2.csv', names=["auto_euclidean", "auto_cosine"])
unrelated_df_2 = pd.read_csv('facenet_dist_unrelated.csv', names=["facenet_euclidean", "facenet_cosine"])
unrelated_df_3 = pd.read_csv('vgg_dist_unrelated.csv', names=["vgg_euclidean", "vgg_cosine"])
unrelated_df = pd.concat([unrelated_df_1, unrelated_df_2, unrelated_df_3], axis=1, sort=False)
unrelated_label = [0] * unrelated_df.shape[0]
unrelated_df['label'] = unrelated_label

train_df = pd.concat([related_df, unrelated_df], axis=0, sort=False)
train_df = train_df.sample(frac=1).reset_index(drop=True)
x = train_df.drop(columns=['label'])
y = train_df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

'''
We have 6 different attributes so max_leaves should be 63, however, setting that number high can cause overfitting, so we set it as 40
'''
params = {
    'objective': 'multiclass',
    'num_class': 2,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 40,
    'verbose': 2
}

model = lgb.train(params, train_data, 500, valid_sets=test_data, early_stopping_rounds=70)


test_df_1 = pd.read_csv('auto_dist_test_v2.csv', names=["auto_euclidean", "auto_cosine"])
test_df_2 = pd.read_csv('facenet_dist_test.csv', names=["facenet_euclidean", "facenet_cosine"])
test_df_3 = pd.read_csv('vgg_dist_test.csv', names=["vgg_euclidean", "vgg_cosine"])
test_df = pd.concat([test_df_1, test_df_2, test_df_3], axis=1, sort=False)

y_pred = model.predict(test_df)
y_submit = [m[1] for m in y_pred]
sub_df = pd.read_csv('sample_submission.csv')
sub_df.is_related = y_submit
sub_df.to_csv("submission_auto_vgg_facenet.csv", index=False)
