'''
Author: Jing Liu
Date: 2019/12/3
'''

'''
This script contains all the methods I have tried before but was not helpful to the final model
'''
#######
# PCA #
#######
# preprocess train data
from imageio import imread
import numpy as np
import os, glob
os.chdir('/train')
from skimage import color

faces = []
count = 1
for file in glob.glob("*/*/*.jpg"):
        img = imread(file)
        im_new = color.rgb2gray(img)
        faces.append(im_new.flatten())

np_faces = np.array(faces)

y = []
curr = 1
for folder in glob.glob("*/*"):
        num = len([name for name in os.listdir(folder)])
        y.extend([curr] * num)
        curr += 1

label = {}
curr = 1
for folder in glob.glob("*/*"):
        vec = folder.split("\\")
        name = vec[0] + '/' + vec[1]
        label[name] = curr
        curr += 1
np_y = np.array(y)

match = []
import pandas as pd
train_df = pd.read_csv("/train_relationships.csv")
train_np = train_df.to_numpy()
for i in range(train_np.shape[0]):
        if train_np[i][0] in label and train_np[i][1] in label:
                p1 = label[train_np[i][0]]
                p2 = label[train_np[i][1]]
                pair = str(p1)+','+str(p2)
                match.append(pair)

# apply PCA on training data
from sklearn.decomposition import PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(np_faces)
X_pca = pca.transform(np_faces)
y_pca = np_y


###############################
# PCA with euclidean distance #
###############################
distances = []
index = []
from scipy.spatial import distance
for i in range(2000):
        for j in range(i+1, 2000):
                if y_pca[i] != y_pca[j]:
                        dist = distance.euclidean(X_pca[i], X_pca[j])
                        distances.append(dist)
                        index.append(str(y_pca[i])+','+str(y_pca[j]))

all_distances = np.array(distances)
sum_dist = np.sum(all_distances)
prob = []
for i in range(len(distances)):
        p = np.sum(all_distances[np.where(all_distances <= all_distances[i])[0]])/sum_dist
        prob.append(1 - p)


######################################
# PCA with Artificial Neural Network #
######################################
'''
This method is for face recognition, not for similar images
'''
X_train, X_test, y_train, y_test = train_test_split(np_faces, np_y, test_size=0.3)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes=(1024,), verbose=True).fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


################################
# Feature Extraction with KAZE #
################################
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from imageio import imread
import _pickle as pickle
import random
import os, glob
import matplotlib.pyplot as plt
files = []
os.chdir('/train')
for folder in glob.glob("*/*"):
	filelist = os.listdir(folder)
	for name in filelist:
		filename = folder + "\\" + name
		files.append(filename)

'''
create KAZE model to get image representation
'''
def extract_features(image_path, vector_size=32):
    image = imread(image_path)
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except Exception as e:
        print ('Error: ', e)
        return None
    return dsc


'''
show extracted images
'''
def show_img(path):
    img = imread(path)
    plt.imshow(img)
    plt.show()


'''
Run the model in batch mode
'''
def batch_extractor(file_path, in_result, pickled_db_path="features.pck"):
	for f in file_path:
		print ('Extracting features from image ', f)
		name = f.split('/')[-1].lower()
		temp = extract_features(f)
		if temp is not None:
			in_result[name] = temp
	with open(pickled_db_path, 'wb') as fp:
		pickle.dump(in_result, fp)


class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

result = {}
batch_extractor(files, result)
ma = Matcher('features.pck')
sample = random.sample(files, 3)
names, match = ma.match(sample[0], topn=30)


################################
# feature matching with OpenCV #
################################
import cv2
import numpy as np
import os
os.chdir('/train')
img1 = cv2.imread("F0002//MID1//P00009_face3.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("F0002//MID2//P00011_face2.jpg", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
lowe_ratio = 0.89
for m,n in matches:
    if m.distance < lowe_ratio*n.distance:
        good.append(m)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:15],None,flags=2)
plt.imshow(img3)
plt.show()
