# KinshipFace
Objective: Given a pair of face images, predict if two people have kinship relationship / MPCS 53112 Final Project

# Data
Train and test data can be downloaded at 
https://www.kaggle.com/c/recognizing-faces-in-the-wild/data
It has two csv files
sample_submission.csv, train_relationship.csv
and train / test image data under
train and test folder
Please note that the program assumes these two csv files and two folders are under the same directory.


# Instructions to run
Kinship_vggpredict.py is for the Convolutional Neural Network Model. You will need to pass the data directory to run the script. Suppose that you store all data under dir /data
then you can use the following command to run
- python3 Kinship_vggpredict.py /data

Kinship_classifier.py is for the Decision Tree Model. Again, you will need to pass the data directory to run the script.
- python3 Kinship_classifier.py /data

Please note that in its intermediate steps, it will output some csv files, so please make sure the working directory has read and write permission. 

The two scripts will output submission_auto_vgg_facenet.csv and submission_vggmodel.csv. You can submit the results in Kaggle to view scores.

Kinship_other.py contains all un-used code like PCA, KAZE, SVM etc. It's not part of the final model.

# Configuration
Before running, please make sure you've installed all the required libraries. You can refer to the import statement in the code.

I collected these code from my Jupyter Notebook, and the only change is filepath. So basically it should not have any error, but if it does, please check all the read / write directories in the code. Also the code may take very long (days or 1-2 weeks) to run, so you can use less image data or change the model parameters to run a simpler version.
